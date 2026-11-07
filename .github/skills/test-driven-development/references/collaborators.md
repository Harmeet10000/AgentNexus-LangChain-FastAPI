# Collaborators

Mock only at system boundaries: external APIs, time or randomness, and, when
needed, databases or the file system. Prefer real in-process collaborators and
a test database when those make the test more representative.

Do not mock classes, modules, or services owned by the application. Verify
behaviour through the public interface instead of treating calls between owned
components as the result.

## Good collaborator boundaries

Inject an external dependency so a test can supply a boundary substitute.

```typescript
function processPayment(order, paymentClient) {
  return paymentClient.charge(order.total);
}
```

Give each boundary a focused operation with a specific input and output shape.

```typescript
const payments = {
  charge: (total) => stripe.charge(total),
};
```

## Bad collaborator boundaries

Do not construct an external client inside the business operation when it
prevents the boundary from being supplied by the caller.

```typescript
function processPayment(order) {
  const client = new StripeClient(process.env.STRIPE_KEY);
  return client.charge(order.total);
}
```

Do not hide unrelated external operations behind one generic fetch function
that requires conditional mock logic. Prefer focused boundary operations such
as `getUser`, `getOrders`, and `createOrder`.
