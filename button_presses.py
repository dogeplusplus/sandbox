
# def cost(values):
#     total = 0
#     previous = 0
#     for v in values:
#         total += abs(v - previous)
#         previous = v

#     return total


def inflate(customers):
    cumulative_cost = 0
    last = 0

    for cust in customers:
        products = sorted(cust)
        # ascending
        if abs(products[0] - last) <= abs(products[-1] - last):
            cumulative_cost += abs(products[0] - last)
            cumulative_cost += products[-1] - products[0]
            last = products[-1]
        else:
            cumulative_cost += abs(products[-1] - last)
            cumulative_cost += products[-1] - products[0]
            last = products[0]

    return cumulative_cost

def inflate2(customers):
    last = 0

    sorted_customers = [sorted(x) for x in customers]

    def recursve(customers, last, current_cost, idx):
        if idx == len(customers):
            return current_cost

        next_customer = customers[idx]
        diff = next_customer[-1] - next_customer[0]

        asc_cost = current_cost + abs(next_customer[0] - last) + diff
        desc_cost = current_cost + abs(next_customer[-1] - last) + diff

        return min(
                recursve(customers, next_customer[0], desc_cost, idx + 1),
                recursve(customers, next_customer[-1], asc_cost, idx + 1)
        )

    return recursve(sorted_customers, last, 0, 0)


if __name__ == "__main__":
    cases = int(input())
    for i in range(cases):
        n, products = [int(x) for x in input().split()]

        customers = []
        for j in range(n):
            prods = [int(x) for x in input().split()]
            customers.append(prods)

        num_presses = inflate(customers)
        print(f"Case #{i+1}: {num_presses}")

