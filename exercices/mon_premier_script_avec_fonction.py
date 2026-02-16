import unittest
from typing import List

MINIMUM_LETTERS = 7


def count_names_with_more_than_seven_letters(first_names: List[str]) -> int:
    names_longer_than_limit = 0

    for first_name in first_names:
        if len(first_name) > MINIMUM_LETTERS:
            names_longer_than_limit += 1
            print(f"{first_name} est un prénom avec un nombre de lettres supérieur à {MINIMUM_LETTERS}")
        else:
            print(f"{first_name} est un prénom avec un nombre de lettres inférieur ou égal à {MINIMUM_LETTERS}")

    return names_longer_than_limit


class TestCountNamesWithMoreThanSevenLetters(unittest.TestCase):

    def test_should_return_number_of_names_longer_than_seven_letters(self) -> None:
        first_names = [
            "Guillaume",
            "Gilles",
            "Juliette",
            "Antoine",
            "François",
            "Cassandre"
        ]

        result = count_names_with_more_than_seven_letters(first_names)

        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
