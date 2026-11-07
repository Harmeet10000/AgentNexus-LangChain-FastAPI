# Patterns and anti-patterns

## Good tests

Test observable behaviour through a public interface. The test should state
what a caller can do, not how the implementation achieves it.

```typescript
test("user can check out a valid cart", async () => {
  const cart = createCart();
  cart.add(product);

  const result = await checkout(cart, paymentMethod);

  expect(result.status).toBe("confirmed");
});
```

Use a known, independent expected value rather than reproducing the
implementation in the assertion.

```typescript
test("calculates the line-item total", () => {
  expect(calculateTotal([{ price: 10 }, { price: 5 }])).toBe(15);
});
```

Use a named parameterized case when cases have the same single behaviour and
only their data differs. Shared setup, teardown, and helpers are useful when
they make that behaviour clearer; do not use them to hide several behaviours
in a single test.

## Bad tests

### Implementation-coupled

Do not test private methods, mock internal collaborators, or assert internal
call sequences. Such tests break during refactors even when observable
behaviour is unchanged.

```typescript
test("checkout calls paymentService.process", async () => {
  const payment = mockPaymentService();
  await checkout(cart, payment);

  expect(payment.process).toHaveBeenCalledWith(cart.total);
});
```

### Self-fulfilling

Do not compute the expected result using the same logic the production code is
supposed to implement. This is circular: the test derives its answer from the
same rule it is meant to check, so it can pass even when that rule is wrong.

```typescript
const expected = items.reduce((sum, item) => sum + item.price, 0);
expect(calculateTotal(items)).toBe(expected);
```

### Horizontal slicing

Do not write a batch of imagined tests and implement them later. Work in
vertical slices: one failing test, the smallest implementation to make it
pass, then the next test. Each test is a tracer bullet that can change what
the next slice should be.
