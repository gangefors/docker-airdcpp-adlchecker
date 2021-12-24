FROM python:3
RUN apt-get update \
  && apt-get install -y libleveldb1d libleveldb-dev \
  && pip install plyvel \
  && rm -rf /var/lib/apt/lists/*
COPY adlchecker.py /usr/bin/adlchecker.py
RUN chmod +x /usr/bin/adlchecker.py
ENTRYPOINT ["/usr/bin/adlchecker.py"]
