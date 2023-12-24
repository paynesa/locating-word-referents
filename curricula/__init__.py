from typing import List, Tuple


def get_curriculum(filepath: str) -> List[Tuple[str, List[str]]]:
    """Generates a list of language-object pairings from a file of the form
    language\n objects\n \n"""
    curriculum: List[Tuple[str, List[str]]] = []
    current_language: str = ""
    i: int = 0
    for line in open(filepath, "r"):
        # this means we have objects, which we add to the curriculum with their language
        if i % 3 == 1:
            objects: List[str] = [object.strip() for object in line.strip().split()]
            curriculum.append((current_language, objects))
        # this means we have language, which we store to be paired with subsequent objects
        elif i % 3 == 0:
            current_language = line.strip()
        i += 1
    return curriculum


def get_verification(filepath: str) -> List[Tuple[str, str]]:
    """Get verification, or gold standard, data to compare the learning output against"""
    verification: List[str, str] = []
    for line in open(filepath, "r"):
        line = line.strip().split()
        verification.append((line[0], line[1]))
    return verification


def load_train_test_curricula() -> Tuple:
    """Loads in the training and testing sets from Stevens et al. 2017"""
    return (
        get_curriculum("curricula/train.txt"),
        get_verification("curricula/train.gold"),
        get_curriculum("curricula/test.txt"),
        get_verification("curricula/test.gold"),
    )


def load_rollins() -> Tuple:
    """Loads in the Rollins file and the gold"""
    return (
        get_curriculum("curricula/rollins.txt"),
        get_verification("curricula/gold.txt")
    )
