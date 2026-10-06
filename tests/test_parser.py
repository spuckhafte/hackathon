"""Tests for the query language.

Two properties matter more than the grammar itself:

* **document/query symmetry** — a query leaf and a document token must normalise
  identically, so `S. 302 IPC` in a query is the `ipc§302` the index holds;
* **nothing raises** — a search box that answers a stray quote with a traceback
  is unusable, so every malformed query must degrade to free text with a warning.

Network-free.  Author: Sumedha (M3)
"""
from __future__ import annotations

import pytest

from nyaya.analyze import analyze_terms
from nyaya.contracts import (
    AndNode,
    FilterNode,
    NotNode,
    OrNode,
    ParsedQuery,
    PhraseNode,
    ProxNode,
    TermNode,
)
from nyaya.query import normalise as N
from nyaya.query.parser import EXAMPLES, Query, describe_tree, explain, parse


def terms(node):
    from nyaya.query.parser import _leaf_terms

    return _leaf_terms(node)


# ==========================================================================
# 1. The contract
# ==========================================================================


def test_parse_returns_a_parsed_query():
    q = parse("privacy")
    assert isinstance(q, ParsedQuery)       # M2's planner takes this type
    assert isinstance(q, Query)             # with warnings bolted on


def test_free_text_takes_the_fast_path():
    q = parse("right to privacy")
    assert q.is_free_text is True
    assert q.root is None
    assert q.free_terms == ["right", "to", "privaci"]
    assert q.filters == []
    assert q.warnings == []


def test_any_operator_leaves_the_fast_path():
    for src in ('privacy AND dignity', '"right to privacy"', 'bail!',
                'murder /3 intention', 'title:(privacy)', 'year:>=2000',
                'privacy NOT bail'):
        q = parse(src)
        assert q.is_free_text is False, src


def test_free_terms_keep_stop_words_but_scoring_terms_drop_them():
    q = parse("right to privacy")
    assert "to" in q.free_terms            # positions must stay exact
    assert "to" not in q.scoring_terms     # but idf(to) ~ 0


def test_empty_and_whitespace_queries():
    for src in ("", "   ", "\n"):
        q = parse(src)
        assert q.is_free_text and q.root is None and q.free_terms == []


# ==========================================================================
# 2. Document / query symmetry
# ==========================================================================


@pytest.mark.parametrize(
    "query,expected",
    [
        ("S. 302 IPC", "ipc§302"),
        ("u/s 302 I.P.C.", "ipc§302"),
        ("Section 302 of the Indian Penal Code, 1860", "ipc§302"),
        ("Article 21 of the Constitution", "const§21"),
        ("Art. 21", "const§21"),
        ("Section 482 CrPC", "crpc§482"),
    ],
)
def test_a_query_statute_normalises_exactly_as_a_document_does(query, expected):
    q = parse(query)
    assert expected in q.free_terms
    # ... and the same token is what analyze() put in the index:
    doc = analyze_terms(f"The appellant was convicted under {query} by the court.")
    assert expected in doc


def test_morphology_matches_between_query_and_document():
    q = parse("convicted")
    assert q.free_terms == ["convict"]
    assert "convict" in analyze_terms("the conviction was upheld")


def test_a_two_word_statute_phrase_collapses_to_the_single_indexed_token():
    """'"section 302"' is two words to a user and one token to the index."""
    q = parse('"section 302" AND murder')
    assert "ipc§302" in terms(q.root)
    assert not any(isinstance(n, PhraseNode) for n in [q.root] + list(q.root.children()))


def test_the_act_fallback_is_query_side_only():
    assert "ipc§302" in parse("sec. 302").free_terms
    # a document saying the same thing with no act in scope gets no guess
    assert "ipc§302" not in analyze_terms("sec. 302")


# ==========================================================================
# 3. Precedence: OR loosest, then AND, then NOT, then proximity
# ==========================================================================


def test_or_is_looser_than_and():
    q = parse("a OR b AND c")
    assert isinstance(q.root, OrNode)
    assert isinstance(q.root.operands[1], AndNode)


def test_and_is_looser_than_not():
    q = parse("a AND b NOT c")
    assert isinstance(q.root, AndNode)
    assert isinstance(q.root.operands[1], NotNode)


def test_not_is_looser_than_proximity():
    q = parse("a /3 b NOT c")
    assert isinstance(q.root, NotNode)
    assert isinstance(q.root.left, ProxNode)


def test_parentheses_override_precedence():
    flat = parse("a OR b AND c")
    grouped = parse("(a OR b) AND c")
    assert isinstance(flat.root, OrNode)
    assert isinstance(grouped.root, AndNode)
    assert isinstance(grouped.root.operands[0], OrNode)


def test_adjacency_is_an_implicit_and_once_the_query_is_boolean():
    q = parse('"due process" dignity autonomy')
    assert isinstance(q.root, AndNode)
    assert len(q.root.operands) == 3


def test_nested_ands_are_flattened_for_the_planner():
    q = parse("a AND b AND c AND d")
    assert isinstance(q.root, AndNode)
    assert len(q.root.operands) == 4        # one n-ary intersection, not a chain


def test_ampersand_and_percent_aliases():
    assert isinstance(parse("a & b").root, AndNode)
    assert isinstance(parse("a % b").root, NotNode)


# ==========================================================================
# 4. The operators
# ==========================================================================


def test_phrase():
    q = parse('"right to privacy"')
    assert isinstance(q.root, PhraseNode)
    assert q.root.terms == ["right", "to", "privaci"]
    assert q.root.raw == "right to privacy"


def test_a_one_word_phrase_degrades_to_a_term():
    assert isinstance(parse('"privacy"').root, TermNode)


@pytest.mark.parametrize(
    "src,kind,k",
    [("a /5 b", "n", 5), ("a w/5 b", "n", 5), ("a /1 b", "n", 1),
     ("a /s b", "s", 0), ("a /p b", "p", 0), ("a /S b", "s", 0)],
)
def test_proximity_operators(src, kind, k):
    node = parse(src).root
    assert isinstance(node, ProxNode)
    assert node.kind == kind and node.k == k


def test_an_absurd_proximity_window_is_clamped_with_a_warning():
    q = parse("a /900 b")
    assert q.root.k == 50
    assert any("wider than a paragraph" in w for w in q.warnings)


def test_root_expansion():
    q = parse("bail!")
    assert isinstance(q.root, TermNode)
    assert q.root.prefix is True
    assert q.root.term == "bail"
    assert parse("bail*").root.prefix is True


def test_a_prefix_root_is_stemmed_because_the_dictionary_holds_stems():
    """A scan for "litigat" finds nothing; the stem "litig" finds litigation."""
    root = parse("litigat!").root.term
    assert root == "litig"          # NOT "litigat", which would match nothing
    for word in ("litigation", "litigant", "litigate"):
        assert analyze_terms(word)[0].startswith(root), word
    # and the general property: the prefix is never longer than the stems it
    # is supposed to find
    for src, words in (("bail!", ("bail", "bailable")),
                       ("privac!", ("privacy",)),
                       ("arbitrat!", ("arbitration", "arbitrator"))):
        pre = parse(src).root.term
        for w in words:
            assert analyze_terms(w)[0].startswith(pre), (src, w)


def test_a_too_short_prefix_is_refused_rather_than_scanning_the_dictionary():
    q = parse("b!")
    assert q.root is None
    assert any("would match most of the dictionary" in w for w in q.warnings)


def test_not_is_a_set_difference_never_a_bare_negation():
    q = parse("privacy NOT bail")
    assert isinstance(q.root, NotNode)
    assert q.root.left.term == "privaci" and q.root.right.term == "bail"


def test_a_leading_not_is_a_friendly_warning_not_a_crash():
    """With the stray NOT dropped the query is just "bail", so it is free text."""
    q = parse("NOT bail")
    assert any("needs something on its left" in w for w in q.warnings)
    assert q.free_terms == ["bail"]
    assert q.is_free_text is True


# ==========================================================================
# 5. Zone restriction
# ==========================================================================


def test_zone_restriction_on_one_word():
    q = parse("title:privacy")
    assert isinstance(q.root, TermNode)
    assert q.root.zone == "title" and q.root.term == "privaci"


def test_zone_restriction_on_a_group_pushes_down_to_every_leaf():
    q = parse("title:(privacy OR dignity)")
    assert isinstance(q.root, OrNode)
    assert [n.zone for n in q.root.operands] == ["title", "title"]


def test_zone_restriction_on_a_phrase():
    q = parse('title:("right to privacy")')
    assert isinstance(q.root, PhraseNode) and q.root.zone == "title"


@pytest.mark.parametrize("zone", ["title", "parties", "statutes", "order", "body"])
def test_every_configured_zone_is_accepted(zone):
    from nyaya import config

    assert zone in config.ZONES
    assert parse(f"{zone}:(privacy)").root.zone == zone


def test_bench_is_a_zone_for_text_and_a_filter_for_a_number():
    """`bench` is the only name that is both; the value decides."""
    numeric = parse("bench:>=5")
    assert numeric.filters == [FilterNode(field="bench", op=">=", value=5)]
    textual = parse("bench:(chandrachud)")
    assert textual.filters == []
    assert textual.root.zone == "bench"


# ==========================================================================
# 6. Filters — must match M2's ParametricIndex schema exactly
# ==========================================================================


def test_filter_fields_match_m2s_domain():
    from nyaya.metadata import PARAMETRIC_FIELDS

    assert set(N.FILTER_FIELDS) == set(PARAMETRIC_FIELDS)


@pytest.mark.parametrize(
    "src,expected",
    [
        ("year:>=2000", FilterNode("year", ">=", 2000)),
        ("year:2000", FilterNode("year", "=", 2000)),
        ("year:<1980", FilterNode("year", "<", 1980)),
        ("bench:>=5", FilterNode("bench", ">=", 5)),
        ("judge:chandrachud", FilterNode("judge", "=", "chandrachud")),
        ("act:NDPS", FilterNode("act", "=", "NDPS")),
        ('type:"criminal appeal"', FilterNode("type", "=", "criminal appeal")),
        ("disposal:allowed", FilterNode("disposal", "=", "allowed")),
    ],
)
def test_filters_are_hoisted_as_conjunctive_restrictions(src, expected):
    q = parse(f"privacy {src}")
    assert q.filters == [expected]
    assert isinstance(q.root, TermNode)      # the filter left the tree


def test_every_filter_in_a_long_query_survives():
    """The bug this pins: a hoisted filter contributes no tree node, and an
    earlier version read that as "nothing there" and dropped the rest."""
    q = parse('privacy year:>=2015 bench:>=5 act:CONST judge:chandrachud '
              'type:"writ petition" disposal:allowed')
    assert [f.field for f in q.filters] == [
        "year", "bench", "act", "judge", "type", "disposal"]


def test_a_filters_only_query_has_no_free_terms():
    q = parse('act:NDPS type:"criminal appeal"')
    assert q.free_terms == []               # not ["act", "ndp", "type", ...]
    assert q.root is None
    assert q.is_free_text is False
    assert len(q.filters) == 2


def test_a_year_range_becomes_two_bounds():
    q = parse("year:2000-2010 AND contempt")
    assert q.filters == [FilterNode("year", ">=", 2000),
                         FilterNode("year", "<=", 2010)]


def test_a_multi_valued_filter_uses_the_in_operator():
    q = parse("act:(IPC OR CRPC)")
    assert q.filters == [FilterNode("act", "in", ["IPC", "CRPC"])]
    assert parse("act:IPC,CRPC").filters == [FilterNode("act", "in", ["IPC", "CRPC"])]


def test_field_aliases_are_accepted():
    assert parse("bench_size:>=5").filters[0].field == "bench"
    assert parse("judges:khanna").filters[0].field == "judge"
    assert parse("case_type:'writ petition'").filters[0].field == "type"


def test_set_values_are_lower_cased_like_m2s_norm_value():
    from nyaya.index.parametric import _norm_value

    q = parse("judge:Chandrachud")
    assert q.filters[0].value == _norm_value("Chandrachud")


def test_act_values_are_upper_cased_to_match_the_canonical_codes():
    assert parse("act:ndps").filters[0].value == "NDPS"


def test_a_filter_inside_parentheses_stays_in_the_tree():
    """Hoisting it would turn an OR into an AND and change the answer."""
    q = parse("privacy AND (year:2000 OR year:2001)")
    assert q.filters == []
    assert isinstance(q.root, AndNode)
    inner = q.root.operands[1]
    assert isinstance(inner, OrNode)
    assert all(isinstance(o, FilterNode) for o in inner.operands)


# ==========================================================================
# 7. Friendly failure — nothing may raise
# ==========================================================================


MALFORMED = [
    'unbalanced "quote',
    "stray ) paren",
    "unclosed ( paren",
    "privacy AND",
    "privacy OR",
    "privacy NOT",
    "privacy /5",
    "/5 privacy",
    "privacy /x dignity",
    "year:recent",
    "year:",
    "bench:many",
    "foo:bar privacy",
    "type:banana",
    "!!!",
    "/",
    '"',
    "()",
    "( ( ( a",
    "a ) ) )",
    "AND OR NOT",
    "title:",
    'title:("',
    "\\ ^ $ [ ] { } * + ?",
    "privacy /0 dignity",
    "act:",
    "year:>=",
]


@pytest.mark.parametrize("src", MALFORMED)
def test_malformed_queries_never_raise(src):
    q = parse(src)                       # must not raise
    assert isinstance(q, Query)
    assert isinstance(q.free_terms, list)
    describe_tree(q.root)                # and the explainer must not raise either
    explain(src)


@pytest.mark.parametrize(
    "src,fragment",
    [
        ('unbalanced "quote', "unbalanced quote"),
        ("stray ) paren", "unbalanced ')'"),
        ("privacy AND", "ends with 'AND'"),
        ("privacy /5", "needs a term on each side"),
        ("privacy /x dignity", "not a proximity operator"),
        ("year:recent", "needs a number"),
        ("foo:bar privacy", "unknown field"),
        ("type:banana", "closed vocabulary"),
        ("NOT bail", "needs something on its left"),
    ],
)
def test_the_warning_says_what_went_wrong(src, fragment):
    q = parse(src)
    assert any(fragment in w for w in q.warnings), q.warnings


def test_an_unknown_field_becomes_ordinary_words_rather_than_vanishing():
    q = parse("foo:bar privacy")
    assert "privaci" in q.free_terms
    assert "bar" in q.free_terms            # the user's words are not thrown away


def test_an_unknown_closed_vocabulary_value_suggests_the_closest():
    q = parse("type:criminal")
    assert any("did you mean 'criminal appeal'" in w for w in q.warnings)


def test_a_dropped_filter_does_not_silently_match_everything():
    q = parse("privacy year:recent")
    assert q.filters == []                  # no bogus filter was invented
    assert any("needs a number" in w for w in q.warnings)


def test_an_out_of_range_year_warns_but_still_filters():
    q = parse("year:1066")
    assert q.filters == [FilterNode("year", "=", 1066)]
    assert any("corpus covers" in w for w in q.warnings)


# ==========================================================================
# 8. Query expansion (query time only)
# ==========================================================================


def test_the_crosswalk_fires_at_query_time():
    q = parse("S. 302 IPC")
    assert q.expansions["ipc§302"] == ["bns§103"]
    assert any("mob-lynching" in n for n in q.notes)


def test_the_crosswalk_runs_in_both_directions():
    assert "ipc§302" in parse("Section 103 of the Bharatiya Nyaya Sanhita").expansions["bns§103"]


def test_a_one_to_many_split_expands_to_both():
    assert parse("Section 498A IPC").expansions["ipc§498a"] == ["bns§85", "bns§86"]


def test_a_bare_number_is_offered_the_penal_code_as_an_addition_not_a_swap():
    q = parse("302 murder")
    assert "302" in q.free_terms             # the original term survives
    assert q.expansions["302"][0] == "ipc§302"


def test_no_expansion_is_invented_where_none_is_verified():
    q = parse("Article 21 of the Constitution")
    assert "const§21" not in q.expansions    # the Constitution was not replaced


# ==========================================================================
# 9. --explain (the demo figure)
# ==========================================================================


def test_describe_tree_draws_a_tree():
    out = describe_tree(parse("a /3 b NOT c AND d").root)
    assert "AND" in out and "NOT" in out and "PROX within 3 words" in out
    assert "|- " in out and "`- " in out
    # every line after the first is indented under its parent
    assert all(line.startswith(("|", "`", " ")) for line in out.splitlines()[1:])


def test_describe_tree_handles_the_free_text_and_filters_only_cases():
    assert "free-text" in describe_tree(None)
    assert "filters only" in explain("act:NDPS")


def test_explain_shows_the_pipeline():
    out = explain("S. 302 IPC AND dying declaration year:>=2000")
    for fragment in ("raw", "canonicalised", "parse tree", "free terms",
                     "tf-idf terms", "filters", "expansions"):
        assert fragment in out


@pytest.mark.parametrize("src", EXAMPLES)
def test_every_documented_example_parses(src):
    explain(src)
