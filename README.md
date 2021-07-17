# Freqtrade-Tester
Test strategies on Freqtrader automatically.

## Add your strategy

Add strategies to the [user_data/strategies](user_data/strategies) folder and also in the [docker-compose.yml](docker-compose.yml) file at `strategy-list` add your strategy in the list.

## Test locally

Install [Docker Compose](https://docs.docker.com/compose/install/).

Run the backtesting command:

```bash
docker-compose run --rm backtesting
```

## Configure run

If you want to change `--max-open-trades` or `--stake-amount` change the [.env](.env) file.