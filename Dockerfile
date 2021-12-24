FROM python:3
RUN apt-get update \
  && apt-get install -y libleveldb1d libleveldb-dev \
  && pip install plyvel \
  && rm -rf /var/lib/apt/lists/*
ADD https://gist.githubusercontent.com/gangefors/9cf33c19f57ec53f7b897ca0680f0f56/raw/ /usr/bin/adlchecker.py
RUN chmod +x /usr/bin/adlchecker.py
ENTRYPOINT ["/usr/bin/adlchecker.py"]
