from typing import List

MIL = int(1e6)

def int_printer(printers: List[List[int]]):
    mins = [MIL, MIL, MIL, MIL]

    for printer in printers:
        mins = [x if x < y else y for (x, y) in zip(mins, printer)]

    if sum(mins) < MIL:
        return "IMPOSSIBLE"
    else:
        for i in range(4):
            mins[i] -= min(mins[i], sum(mins) - MIL)
            if sum(mins) == MIL:
                return " ".join([str(x) for x in mins])


if __name__ == "__main__":
    cases = int(input())
    for i in range(cases):
        printers = []
        for p in range(3):
            inks = [int(x) for x in input().split(" ")]
            printers.append(inks)

        ink_combination = int_printer(printers)
        print(f"Case #{i+1}: {ink_combination}")

