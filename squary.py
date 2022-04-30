def squary(nums, k):
    if k == 1:
        sm = sum(nums)
        smsq = sum([x**2 for x in nums])

        if sm == 0:
            if smsq == 0:
                return 1
            return "IMPOSSIBLE"

        x = (smsq - sm ** 2) / (2 * sm)
        if x.is_integer():
            return int(x)
        else:
            return "IMPOSSIBLE"
    else:
        sm = sum(nums)
        smsq = sum([x**2 for x in nums])

        while k > 0:



if __name__ == "__main__":
    cases = int(input())
    for i in range(cases):
        n, k = [int(x) for x in input().split()]
        nums = [int(x) for x in input().split()]

        is_squary = squary(nums, k)

        print(f"Case #{i+1}: {is_squary}")
