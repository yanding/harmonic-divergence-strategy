# Freqtrade-Tester
Test strategies on Freqtrader automatically.

Testing [NFI Next 6726d91](https://github.com/iterativv/NostalgiaForInfinity/commit/6726d91178fb07cbd8d4294c791d06b5015ee5ae)

## Add your strategy

Add strategies to the [user_data/strategies](user_data/strategies) folder and also in the [docker-compose.yml](docker-compose.yml) file at `strategy-list` add your strategy in the list.

## Test locally

Install [Docker Compose](https://docs.docker.com/compose/install/).

Run the backtesting command:

```bash
docker-compose run --rm backtesting
```

## Configure run

If you want to change `--max-open-trades` or `--stake-amount` or `--timerange` change the [.env](.env) file.


## Update pairs or timeframe

If you want to update pairs [user_data/pairlists.json](user_data/pairlists.json) from `exchange:pair_whitelist` or timeframe from [docker-compose.yml](docker-compose.yml) from `download-data:timerange`, run the following after you changed.

```bash
docker-compose run --rm download-data
```