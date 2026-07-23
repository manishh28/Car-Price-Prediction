import unittest

from vehicle_search import normalize_search, rank_matches


class VehicleSearchTests(unittest.TestCase):
    def test_normalizes_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_search("Mercedes-Benz  B180"), "mercedes benz b180")

    def test_exact_match_ranks_first(self) -> None:
        options = ["Toyota Fortuner 2.5", "Toyota Fortuner", "Toyota Etios"]
        self.assertEqual(rank_matches(options, "Toyota Fortuner")[0], "Toyota Fortuner")

    def test_matches_model_word_without_brand_prefix(self) -> None:
        options = ["BMW 3 Series 320d", "BMW 6 Series GT 630d", "BMW X1"]
        self.assertEqual(rank_matches(options, "3 Series"), ["BMW 3 Series 320d"])

    def test_frequency_breaks_equally_relevant_ties(self) -> None:
        options = ["Toyota Fortuner A", "Toyota Fortuner B"]
        counts = {"Toyota Fortuner A": 2, "Toyota Fortuner B": 8}
        self.assertEqual(
            rank_matches(options, "Fortuner", counts)[0],
            "Toyota Fortuner B",
        )

    def test_empty_query_has_no_matches(self) -> None:
        self.assertEqual(rank_matches(["Toyota"], ""), [])


if __name__ == "__main__":
    unittest.main()
