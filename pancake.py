
def pancake(stack):
    left = 0
    right = len(stack) - 1

    num_payers = 0
    deliciousness = -1
    while left <= right:
        if stack[left] < stack[right]:
            if stack[left] >= deliciousness:
                num_payers += 1
                deliciousness = stack[left]
            left += 1
        else:
            if stack[right] >= deliciousness:
                num_payers += 1
                deliciousness = stack[right]
            right -= 1

    return num_payers


if __name__ == "__main__":
    cases = int(input())
    for i in range(cases):
        n_pancakes = int(input())
        stack = [int(x) for x in input().split()]

        num_payers = pancake(stack)
        print(f"Case #{i+1}: {num_payers}")

