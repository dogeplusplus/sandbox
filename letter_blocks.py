def megatower(towers):
    substacks = []
    substack = towers[0]

    remaining = towers[1:]
    while len(remaining) > 0:
        left_candidates = [x for x in remaining if x[0] == substack[-1]]

        if len(left_candidates) > 0:
            singular_letter = [x for x in left_candidates if len(set(x)) == 1]
            if len(singular_letter) > 0:
                chosen = singular_letter[0]
            else:
                chosen = left_candidates[0]

            substack = substack + chosen
            remaining.remove(chosen)

        right_candidates = [x for x in remaining if x[-1] == substack[0]]

        if len(right_candidates) > 0:
            singular_letter = [x for x in right_candidates if len(set(x)) == 1]
            if len(singular_letter) > 0:
                chosen = singular_letter[0]
            else:
                chosen = right_candidates[0]

            substack = chosen + substack
            remaining.remove(chosen)

        if len(left_candidates) == 0 and len(right_candidates) == 0:
            substacks.append(substack)
            substack = remaining[0]
            remaining = remaining[1:]

    if len(substack) > 0:
        substacks.append(substack)

    result = ""
    for tower in substacks:
        if len(set(tower).intersection(set(result))) > 0:
            return "IMPOSSIBLE"
        elif not dutch_flag(tower):
            return "IMPOSSIBLE"
        else:
            result = result + tower

    return result

def dutch_flag(tower):
    seen_letters = set(tower[0])
    previous_letter = tower[0]

    for letter in tower[1:]:
        if letter in seen_letters and letter != previous_letter:
            return False
        elif letter not in seen_letters:
            previous_letter = letter
            seen_letters.add(letter)
    return True


if __name__ == "__main__":
    cases = int(input())
    for i in range(cases):
        num_towers = int(input())
        towers = input().split(" ")

        mega = megatower(towers)
        print(f"Case #{i+1}: {mega}")
