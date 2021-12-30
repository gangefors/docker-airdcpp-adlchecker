#!/usr/bin/env python3
"""
ADL search for AirDC++w (https://airdcpp-web.github.io)

Requires:
  Python 3.7+ or 3.6 with dataclasses (https://pypi.org/project/dataclasses/)

Optional:
  plyvel, for faster scanning using the FileIndex DB instead of checking
          files on disk. Linux only, read install instructions thoroughly.
          https://plyvel.readthedocs.io/en/latest/installation.html

Usage:
  ./adlsearch.py --help

  This script can also be run using Docker:
    docker run --rm --volumes-from <airdcpp-container> gangefors/airdcpp-adlchecker --help
"""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import argparse
import contextlib
import hashlib
import pathlib
import re
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as et
from argparse import Namespace
from dataclasses import dataclass
from glob import glob
from itertools import groupby
from os import path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

_FILE_INDEX_DB = None
_PATH_SIZE_CACHE: Dict[str, int] = {}


@dataclass
class ADLRule:
    directory: str
    max_size: int
    min_size: int
    regex: Any
    source_type: str


@dataclass
class Cache:
    path: str
    xml: Any


@dataclass
class Config:
    no_ignore: bool
    path: str
    print_progress: bool
    profile: str


@dataclass
class Profile:
    name: str
    paths: List[str]


@dataclass
class Share:
    directory: str
    name: str
    path: str
    type: str


@dataclass
class Match:
    adl_item: ADLRule
    share_item: Share


def main(config: Config):
    start = timer()

    print("Loading profiles...", end="", flush=True)
    profile = select_profile(config)
    print("OK!")

    print("Loading ADL...", end="", flush=True)
    adl = get_adl_items(config)
    print("OK!")

    print("Loading share '{}'...".format(profile.name), end="", flush=True)
    caches = get_share_caches(config)
    share = get_share_items(profile, caches)
    print("OK!")

    print("Searching...", end="", flush=True)
    matches = perform_adl_search(adl, share, config)
    filtered = filter_matches(config, matches) if not config.no_ignore else matches
    if not config.print_progress:
        print("OK!")
    print("> ADL:          {} entries".format(len(adl)))
    print("> Share:        {} entries".format(len(share)))
    print("> Matched:      {} entries".format(len(matches)))
    print("> Ignored:      {} entries".format(len(matches) - len(filtered)))
    print("> Time spent:   {:.2f} seconds".format(timer() - start))

    if len(filtered) > 0:
        print("> Results:")
        print_matches(filtered)


# CLI / config
def get_config(args: Namespace) -> Config:
    config_path = first(args.config) if args.config else find_config_path()

    if not config_path:
        print("Error!")
        throw_error("Configuration not found.")

    return Config(
        path=config_path,
        profile=first(args.profile),
        no_ignore=args.no_ignore,
        print_progress=args.progress,
    )


def get_cli_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="Perform an ADL search.")
    parser.add_argument(
        "--config", nargs=1, default=None, help="Path to the AirDC++ config directory"
    )
    parser.add_argument(
        "--profile",
        nargs=1,
        default=None,
        help="The share profile to run the ADL against",
    )
    parser.add_argument(
        "--no-ignore",
        action="store_true",
        default=False,
        help="Do not hide results from ADLSearch.ignorelist",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Print progress while scanning.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        default=False,
        help="Don't use the FileIndex DB, instead stat files on fs.",
    )

    return parser.parse_args()


def find_config_path() -> Optional[str]:
    return coalesce([find_config_in_bootfile(), find_config_from_default()])


def find_config_in_bootfile() -> Optional[str]:
    if not path.isfile("dcppboot.xml"):
        return None

    element = et.parse("dcppboot.xml").getroot().find("./ConfigPath")
    return replace_bootfile_vars(element.text) if element is not None else None


def replace_bootfile_vars(text: Optional[str]) -> Optional[str]:
    return text.replace("%[HOME]", path.expanduser("~/")) if text else None


def find_config_from_default() -> Optional[str]:
    dirs = [".", path.expanduser("~/.airdc++"), "./config", "/.airdcpp"]
    return first([d for d in dirs if path.isfile(path.join(d, "DCPlusPlus.xml"))])


# ADL result
def print_matches(matches: List[Match]):
    matches = sorted(matches, key=lambda x: x.adl_item.directory)
    for key, group in groupby(matches, lambda x: x.adl_item.directory):
        print(key)
        print("\n".join([x.share_item.path for x in group]), flush=True)


def print_progress(progress: int, total: int):
    if progress % 1000 == 0:
        print(
            f"\rSearching...{progress/total*100:>5.1f}% "
            f"{int(progress/1000)}k/{int(total/1000)}k",
            end="",
            flush=True,
        )
    if progress == total:
        print("\x1b[2K", end="\r")
        print("Searching...OK!")


# ADL matching
def perform_adl_search(
    adl: List[ADLRule], share: List[Share], config: Config
) -> List[Match]:
    matches: List[Match] = []
    total = len(adl) * len(share)
    for i, match in enumerate(
        (Match(adl_item, share_item) for adl_item in adl for share_item in share), 1
    ):
        if config.print_progress:
            print_progress(i, total)
        if is_adl_match(match.adl_item, match.share_item):
            matches.append(match)
    return matches


def is_adl_match(adl_item: ADLRule, share_item: Share) -> bool:
    return (
        is_source_match(adl_item, share_item)
        and is_regex_match(adl_item, share_item)
        and is_size_match(adl_item, share_item)
    )


def is_source_match(adl_item: ADLRule, share_item: Share) -> bool:
    if adl_item.source_type == "Filename" and share_item.type == "File":
        return True
    if adl_item.source_type == "Directory" and share_item.type == "Directory":
        return True
    if adl_item.source_type == "Full Path":
        return True
    return False


def is_regex_match(adl_item: ADLRule, share_item: Share) -> bool:
    text = share_item.path if adl_item.source_type == "Full Path" else share_item.name

    return adl_item.regex.search(text) is not None


def is_size_match(adl_item: ADLRule, share_item: Share) -> bool:
    if share_item.type == "Directory":
        return True

    min_size, max_size = adl_item.min_size, adl_item.max_size
    if _FILE_INDEX_DB:
        # Use FileIndex DB if available, faster than checking files on disk.
        # The DB stores a struct for each file consisting of the following values:
        #   version: int8 (1 Byte)
        #   timestamp: int64 (8 Byte)
        #   TTH: 192 bits (24 Bytes)
        #   filesize: int64 (8 Bytes)
        file_size = struct.unpack(
            "<bq24sq", _FILE_INDEX_DB.get(share_item.path.lower().encode("utf-8"))
        )[-1]
    else:
        file_size = getsize_proxy(share_item.path)

    return (
        file_size > -1
        and (min_size == -1 or file_size >= min_size)
        and (max_size == -1 or file_size <= max_size)
    )


def getsize_proxy(full_path: str) -> int:
    global _PATH_SIZE_CACHE  # pylint: disable=global-statement
    hashed_path = hashlib.md5(full_path.encode("utf-8")).hexdigest()

    if _PATH_SIZE_CACHE.get(hashed_path, False):
        return _PATH_SIZE_CACHE[hashed_path]

    if not path.isfile(full_path):
        _PATH_SIZE_CACHE[hashed_path] = -1
        return -1

    _PATH_SIZE_CACHE[hashed_path] = path.getsize(full_path)
    return _PATH_SIZE_CACHE[hashed_path]


# Ignore list
def filter_matches(config: Config, matches: List[Match]) -> List[Match]:
    file_path = path.join(config.path, "ADLSearch.ignorelist")

    if not path.isfile(file_path):
        return matches

    with open(file_path, encoding="utf-8") as filep:
        ignore_list = filep.read().splitlines()

    return [
        match
        for match in matches
        if not any(
            (
                match.share_item.path.startswith(ln)
                for ln in ignore_list
                if ln and not ln.startswith("#")
            )
        )
    ]


# ADL search items
def get_adl_items(config: Config) -> List[ADLRule]:
    xml = get_xml_config(config, "ADLSearch.xml")
    parsed = [parse_adl_item(e) for e in xml.findall("./SearchGroup/Search")]

    return [it for it in parsed if it is not None]


def parse_adl_item(element: et.Element) -> Optional[ADLRule]:
    search: str = element.find("./SearchString").text
    dest: str = element.find("./DestDirectory").text
    is_regex: str = element.find("./SearchString").attrib["RegEx"]
    source_type: str = element.find("./SourceType").text
    is_active: str = element.find("./IsActive").text
    size_type: str = element.find("./SizeType").text
    min_size: str = element.find("./MinSize").text
    max_size: str = element.find("./MaxSize").text

    def get_byte_size(size_text: str, size_type: str):
        value = int(size_text)

        if value < 0:
            return -1
        if size_type.lower() == "mib":
            return value * 1024 * 1024
        if size_type.lower() == "kib":
            return value * 1024
        if size_type.lower() == "gib":
            return value * 1024 * 1024 * 1024
        return value

    if is_regex == "1" and is_active == "1":
        return ADLRule(
            source_type=source_type,
            regex=re.compile(unboost_regex(search), flags=re.IGNORECASE),
            directory=dest,
            min_size=get_byte_size(min_size, size_type),
            max_size=get_byte_size(max_size, size_type),
        )

    return None


# Profiles
def get_share_profiles(config: Config) -> List[Profile]:
    xml = get_xml_config(config, "DCPlusPlus.xml")
    elements = [
        e for e in xml.findall("./*[@Name]") if e.tag in ["Share", "ShareProfile"]
    ]

    return [it for it in [parse_share_profile(e) for e in elements] if it is not None]


def parse_share_profile(element: et.Element) -> Profile:
    return Profile(
        name=element.attrib["Name"],
        paths=[e.text for e in element.findall("./Directory") if e.text],
    )


def select_profile(  # pylint: disable=inconsistent-return-statements
    config: Config,
) -> Profile:
    profiles = get_share_profiles(config)

    if len(profiles) == 0:
        throw_error("No share profiles found.")
    elif len(profiles) == 1:
        return profiles[0]
    elif not config.profile:
        print("Error!")
        print(f"Multiple profiles found: {', '.join([p.name for p in profiles])}")
        print("Select the right one with --profile PROFILE")
        sys.exit(-1)
    else:
        matches = [p for p in profiles if p.name.lower() == config.profile.lower()]

        if not matches:
            throw_error(f"Profile '{config.profile}' not found.")

        return matches[0]


# Share
def get_share_caches(config: Config) -> List[Cache]:
    xmlglob = path.join(config.path, "ShareCache", "*.xml")

    return [get_share_cache(xml) for xml in glob(xmlglob)]


def get_share_cache(cache_path: str) -> Cache:
    xml = et.parse(cache_path).getroot()

    return Cache(path=xml.attrib["Path"], xml=xml)


def get_share_items(profile: Profile, caches: List[Cache]) -> List[Share]:
    my_caches = [c for c in caches if any((c.path == p for p in profile.paths))]

    return [it for c in my_caches for it in expand_share_items(c)]


def expand_share_items(cache: Cache) -> List[Share]:
    items: List[Share] = []

    def expand(element: et.Element, current_dir: str):
        for elm in [
            name
            for name in element.findall("./*[@Name]")
            if name.tag in ["File", "Directory"]
        ]:
            item = parse_share_item(elm, current_dir)
            items.append(item)

            if item.type == "Directory":
                expand(elm, item.path)

    expand(cache.xml, cache.path)

    return items


def parse_share_item(element: et.Element, current_dir: str) -> Share:
    item_name: str = element.attrib["Name"]

    return Share(
        type=element.tag,
        name=item_name,
        directory=current_dir,
        path=path.join(current_dir, item_name),
    )


# Tools
def get_xml_config(config: Config, filename: str) -> et.ElementTree:
    full_path = path.join(config.path, filename)

    if not path.isfile(full_path):
        print("Error!")
        throw_error(f"Config file '{full_path}' not found")

    return et.parse(full_path)


def coalesce(lst: List[Any]) -> Any:
    return first([it for it in lst if it])


def first(lst: List[Any]) -> Any:
    return (lst or [None])[0]


def throw_error(message: str):
    print(message)
    sys.exit(-1)


def unboost_regex(pattern: str) -> str:
    py_regex = pattern.replace("(?-i)", "").replace("(?i)", "")

    if py_regex.startswith("(?i:"):
        py_regex = "(?:" + py_regex[4:]

    return py_regex


@contextlib.contextmanager
def file_info_db(path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Loading FileIndex DB...", end="", flush=True)
        dest = pathlib.Path(tmpdir).joinpath("FileIndex")
        shutil.copytree(
            pathlib.Path(f"{path}/FileIndex/"),
            dest,
            ignore=shutil.ignore_patterns("LOCK"),
        )
        print("OK!")
        yield dest


if __name__ == "__main__":
    args = get_cli_arguments()
    config = get_config(args)
    if args.no_db:
        main(config)
    else:
        try:
            import plyvel

            with file_info_db(config.path) as db_path:
                _FILE_INDEX_DB = plyvel.DB(str(db_path))
                main(config)
        except ImportError:
            print("Can't load FileIndex DB, falling back to filesystem.")
            main(config)
        finally:
            if _FILE_INDEX_DB and not _FILE_INDEX_DB.closed:
                _FILE_INDEX_DB.close()
