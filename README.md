# Freqtrade-Tester
Test strategies on Freqtrader automatically.

## Add your strategy

Add strategies to the `user_data/strategies` folder, create a Pull Request into `main` and see the result or test it locally.

## Test locally

Install [Docker Compose](https://docs.docker.com/compose/install/).

Run the backtesting command:

```bash
docker-compose run --rm backtesting
```