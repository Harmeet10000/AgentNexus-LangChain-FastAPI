-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create application database if not exists
SELECT 'Database initialized' AS status;
