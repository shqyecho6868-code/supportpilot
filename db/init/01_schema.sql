-- SupportPilot schema. Executed by the Postgres entrypoint on first boot,
-- and by `scripts/reset_db.py` / the test suite to rebuild a clean database.

CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS orders (
    id               TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers(id),
    status           TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    items            JSONB NOT NULL,
    total_cents      INTEGER NOT NULL CHECK (total_cents >= 0),
    currency         TEXT NOT NULL DEFAULT 'USD',
    shipping_address JSONB NOT NULL,
    carrier          TEXT,
    tracking_number  TEXT,
    placed_at        TIMESTAMPTZ NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);

-- Append-only audit trail of every state change made through the tools.
CREATE TABLE IF NOT EXISTS order_events (
    id          BIGSERIAL PRIMARY KEY,
    order_id    TEXT NOT NULL REFERENCES orders(id),
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_order_events_order ON order_events(order_id);

-- Idempotency records for mutating tools. Keys are scoped per user so one
-- customer can never replay (or collide with) another customer's key.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    user_id       TEXT NOT NULL REFERENCES customers(id),
    key           TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    response      JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS tickets (
    id           TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(id),
    order_id     TEXT REFERENCES orders(id),
    subject      TEXT NOT NULL,
    description  TEXT NOT NULL,
    priority     TEXT NOT NULL DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high')),
    status       TEXT NOT NULL DEFAULT 'open',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE SEQUENCE IF NOT EXISTS ticket_seq START 1001;
