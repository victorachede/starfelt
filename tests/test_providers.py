from starfelt.providers.base import pick_cheapest


def test_pick_cheapest():
    offer = pick_cheapest(preferred=["runpod", "lambda"], allow_spot=True)
    assert offer.usd_per_hour > 0
    assert offer.name in {"runpod", "lambda", "aws", "gcp"}
