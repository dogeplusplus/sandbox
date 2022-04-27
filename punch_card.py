def punch_card(rows, cols):
    even = ["+", "-"]
    odd = ["|", "."]

    even_end = "+"
    odd_end = "|"

    for row in range(rows * 2 + 1):
        if row % 2 == 0:
            row_ascii = cols * even + [even_end]
        else:
            row_ascii = cols * odd + [odd_end]

        if row <= 1:
            row_ascii[0] = "."
            row_ascii[1] = "."
        print("".join(row_ascii))


if __name__ == "__main__":
    cases = int(input())
    for i in range(cases):
        rows, cols = [int(x) for x in input().split(" ")]

        print(f"Case #{i+1}:")
        punch_card(rows, cols)
