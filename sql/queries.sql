-- The SQL behind the dashboard (detector/stats.py), runnable as-is in sqlite3 or psql.
-- Portable SQL only: JOIN, LEFT JOIN, GROUP BY, CASE, COALESCE, DATE.

-- 1. Headline numbers
SELECT COUNT(*)                                                                 AS total,
       COALESCE(AVG(confidence), 0)                                             AS avg_confidence,
       COALESCE(SUM(CASE WHEN uncertain THEN 1 ELSE 0 END), 0)                  AS uncertain,
       COALESCE(SUM(CASE WHEN true_disease_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS reviewed,
       COALESCE(SUM(CASE WHEN true_disease_id = predicted_disease_id THEN 1 ELSE 0 END), 0) AS confirmed,
       COUNT(DISTINCT user_id)                                                  AS users
FROM detector_prediction;

-- 2. Diagnoses per disease (LEFT JOIN keeps diseases with no predictions yet)
SELECT d.name,
       COUNT(p.id)                                               AS predictions,
       AVG(p.confidence)                                         AS avg_confidence,
       COALESCE(SUM(CASE WHEN p.uncertain THEN 1 ELSE 0 END), 0) AS uncertain
FROM detector_disease d
LEFT JOIN detector_prediction p ON p.predicted_disease_id = d.id
GROUP BY d.id, d.name, d.display_order
ORDER BY d.display_order, d.name;

-- 3. Activity per day (UTC) since a given date
SELECT DATE(created_at) AS day, COUNT(*) AS n
FROM detector_prediction
WHERE created_at >= '2026-01-01'
GROUP BY DATE(created_at)
ORDER BY day;

-- 4. Feedback confusion matrix: what the model said vs. what users confirmed
SELECT pd.name AS predicted, td.name AS actual, COUNT(*) AS n
FROM detector_prediction p
JOIN detector_disease pd ON pd.id = p.predicted_disease_id
JOIN detector_disease td ON td.id = p.true_disease_id
GROUP BY pd.name, td.name;

-- 5. Review queue: low-confidence diagnoses nobody has checked yet
SELECT p.id, d.name AS predicted, ROUND(CAST(p.confidence AS numeric), 3) AS confidence, p.created_at
FROM detector_prediction p
JOIN detector_disease d ON d.id = p.predicted_disease_id
WHERE p.uncertain AND p.true_disease_id IS NULL
ORDER BY p.created_at DESC
LIMIT 20;

-- 6. Most active users
SELECT u.username, COUNT(p.id) AS diagnoses, MAX(p.created_at) AS last_diagnosis
FROM auth_user u
JOIN detector_prediction p ON p.user_id = u.id
GROUP BY u.id, u.username
ORDER BY diagnoses DESC;
