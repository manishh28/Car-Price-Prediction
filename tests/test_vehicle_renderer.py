import csv
from collections import defaultdict
from pathlib import Path
import unittest

from vehicle_renderer import (
    BODY_STYLES,
    PAINT_COLOR,
    render_vehicle_preview,
    vehicle_body_style,
    vehicle_shape_seed,
)


DATA_PATH = Path(__file__).parents[1] / "car details.csv"


def load_models() -> dict[str, list[dict[str, str]]]:
    models: dict[str, list[dict[str, str]]] = defaultdict(list)
    with DATA_PATH.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("name"):
                models[row["name"]].append(row)
    return models


class VehicleRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = load_models()

    def test_every_database_model_has_local_3d_configuration(self) -> None:
        seeds: set[int] = set()

        for model, records in self.models.items():
            seat_values = sorted(
                int(float(row["seats"])) for row in records if row.get("seats")
            )
            seats = seat_values[len(seat_values) // 2] if seat_values else 5
            body_style = vehicle_body_style(model, seats)
            document = render_vehicle_preview(
                brand=model.split()[0],
                model=model,
                year=2018,
                fuel="Petrol",
                transmission="Manual",
                matching_records=len(records),
                seats=seats,
            )

            self.assertIn(body_style, BODY_STYLES)
            self.assertIn(model, document)
            self.assertGreaterEqual(document.count(PAINT_COLOR), 2)
            self.assertNotIn("sketchfab", document.lower())
            self.assertNotIn("wikimedia", document.lower())
            seeds.add(vehicle_shape_seed(model))

        self.assertEqual(len(self.models), 2_058)
        self.assertEqual(len(seeds), len(self.models))

    def test_reported_models_use_expected_body_styles(self) -> None:
        expectations = {
            "BMW 3 Series 320d": "sedan",
            "BMW X1 sDrive20d": "suv",
            "Land Rover Freelander 2 TD4 HSE": "suv",
            "Toyota Etios V": "sedan",
            "Toyota Fortuner 2.5 4x2 MT TRD Sportivo": "suv",
            "Toyota Innova 2.5 EV (Diesel) PS 7 Seater BS IV": "mpv",
        }

        for model, expected_style in expectations.items():
            seats = 7 if expected_style == "mpv" else 5
            self.assertEqual(vehicle_body_style(model, seats), expected_style)

    def test_body_paint_is_globally_consistent(self) -> None:
        self.assertEqual(PAINT_COLOR, "#18786d")


if __name__ == "__main__":
    unittest.main()
