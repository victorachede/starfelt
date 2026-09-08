import warnings

from starfelt.providers.base import PRICES_LAST_UPDATED, pick_cheapest


def test_pick_cheapest_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        offer = pick_cheapest(preferred=["runpod", "lambda"], allow_spot=True)
        assert offer.usd_per_hour > 0
        assert any("estimates" in str(x.message).lower() for x in w)
    assert PRICES_LAST_UPDATED
