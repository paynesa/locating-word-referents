from typing import List, Tuple

def get_curriculum(filepath : str)->List[Tuple[str, List[str]]]:
    """Generates a list of language-object pairings from a file of the form
    language\n objects\n \n"""
    curriculum : List[Tuple[str, List[str]]] = []
    current_language : str = ""
    i : int = 0
    for line in open(filepath, "r"):
        # this means we have objects, which we add to the curriculum with their language
        if i%3 == 1:
            objects : List[str] = [object.strip() for object in line.strip().split()]
            curriculum.append((current_language, objects))
        # this means we have language, which we store to be paired with subsequent objects
        elif i%3 == 0:
            current_language = line.strip()
        i+=1
    return curriculum






