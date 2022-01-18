# AirDC++w ADL checker

Scan your share with ADLSearch using the included Python script.

You need to add an ADLSearch.xml file to the AirDC++w config directory
for the script to work.

If you are running AirDC++w using the gangefors/airdcpp-webclient Docker
image, then running the script using Docker is recommended.
Run the following command in a terminal on the same host as you are
running the AirDC++w container.

    docker run --rm --volumes-from <airdcpp-container-name-or-id> \
        gangefors/airdcpp-adlchecker --help

Or you can run the script in a terminal using Python 3.7+, check script
file for additional info.

    python3 adlchecker.py --help
