-- Example query for a SQL database containing synthetic_driver_data
SELECT region, fleet_size, COUNT(*) AS drivers,
       AVG(CAST(future_12m_incident AS FLOAT)) AS incident_rate,
       AVG(prior_violations) AS avg_prior_violations
FROM synthetic_driver_data
GROUP BY region, fleet_size
ORDER BY incident_rate DESC;
