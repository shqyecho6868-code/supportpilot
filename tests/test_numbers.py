"""The deterministic number check that runs next to the LLM task judge."""

from supportpilot.evals.numbers import check_numbers, quantities

POLICY = (
    "Refunds are issued within 3 business days after the item passes inspection, which takes up to "
    "2 business days. Card refunds then take 5 to 10 business days to post."
)


def test_quantities_parse_units_ranges_words_and_money():
    text = "A 15-day window, 5 to 10 business days, two years, $1,000 or $6.95, a 15% fee, 48 hours."
    assert quantities(text) == {
        (15, "day"),
        (5, "day"),
        (10, "day"),
        (2, "year"),
        (1000, "$"),
        (6.95, "$"),
        (15, "%"),
        (48, "hour"),
    }


def test_citations_and_order_ids_are_not_quantities():
    assert quantities("See [Returns Policy § 30-day window]. ORD-1001 ships to CA 94105.") == set()


def test_correct_answer_passes():
    reply = "Inspected within 2 business days, refunded within 3, and your bank takes 5-10 business days."
    check = check_numbers(reply, ["2 business days", "3 business days", "5 to 10 business days"], [POLICY])
    assert check.ok, check


def test_computed_total_is_flagged_as_unsupported():
    """The P3 failure the judge passed: a total the policy never states (and that is wrong)."""
    reply = "In total it takes roughly 7 to 13 business days."
    check = check_numbers(reply, [], [POLICY])
    assert check.unsupported == ("7 day", "13 day")
    assert not check.ok


def test_missing_expected_number_is_flagged():
    check = check_numbers("Most items can be returned within 30 days.", ["30 days", "15 days"], [POLICY])
    assert check.missing == ("15 day",)


def test_expected_value_without_its_unit_counts_as_stated():
    reply = "You have 30 days from delivery (60 for Plus members)."
    assert check_numbers(reply, ["30 days", "60 days"], []).present


def test_numbers_from_the_conversation_are_allowed():
    """Echoing the customer's own numbers ("40 days ago") is not an unsupported claim."""
    reply = "Even though it's been 40 days, exchanges are allowed within 45 days."
    check = check_numbers(reply, ["45 days"], ["exchanges within 45 days", "I got a sweater 40 days ago"])
    assert check.ok, check


def test_amounts_from_tool_results_in_usd_support_dollar_figures_in_the_reply():
    result = check_numbers("ORD-1001 totals $129.99.", [], ['{"total": "129.99 USD"}'])
    assert result.supported
