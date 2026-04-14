BEGIN TRANSACTION;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(200) UNIQUE
);

SELECT * FROM users WHERE active = true;

SELECT u.name, u.email, COUNT(o.id) as order_count
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.active = $1
GROUP BY u.name, u.email
ORDER BY order_count DESC;

DELETE FROM logs;

INSERT INTO users (name, email) VALUES (?, ?);

SELECT * FROM (
    SELECT * FROM (
        SELECT id FROM users WHERE score > (
            SELECT AVG(score) FROM users
        )
    ) sub1
) sub2;

DROP TABLE IF EXISTS temp_data;

COMMIT;
