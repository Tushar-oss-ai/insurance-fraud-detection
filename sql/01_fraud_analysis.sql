
-- Overall fraud snapshot
SELECT COUNT(*) AS total_claims, SUM(FraudFound_P) AS fraud_claims,
       ROUND(100.0 * AVG(FraudFound_P), 2) AS fraud_rate_pct FROM claims;

-- H3: Fraud rate by policy type
SELECT PolicyType, COUNT(*) AS n_claims, SUM(FraudFound_P) AS n_fraud,
       ROUND(100.0 * AVG(FraudFound_P), 2) AS fraud_rate_pct
FROM claims GROUP BY PolicyType HAVING COUNT(*) >= 50 ORDER BY fraud_rate_pct DESC;

-- H4: Fraud rate by fault and police report
SELECT Fault, PoliceReportFiled, COUNT(*) AS n_claims,
       ROUND(100.0 * AVG(FraudFound_P), 2) AS fraud_rate_pct
FROM claims GROUP BY Fault, PoliceReportFiled ORDER BY fraud_rate_pct DESC;

-- Fraud rate by address change before claim
SELECT AddressChange_Claim, COUNT(*) AS n_claims,
       ROUND(100.0 * AVG(FraudFound_P), 2) AS fraud_rate_pct
FROM claims GROUP BY AddressChange_Claim ORDER BY fraud_rate_pct DESC;

-- H5: Fraud rate by past claims history
SELECT PastNumberOfClaims, PastNumberOfClaims_ord, COUNT(*) AS n_claims,
       ROUND(100.0 * AVG(FraudFound_P), 2) AS fraud_rate_pct
FROM claims GROUP BY PastNumberOfClaims, PastNumberOfClaims_ord ORDER BY PastNumberOfClaims_ord;

-- Fraud rank by vehicle make (window function)
WITH make_stats AS (
    SELECT Make, COUNT(*) AS n_claims, SUM(FraudFound_P) AS n_fraud,
           100.0 * AVG(FraudFound_P) AS fraud_rate_pct,
           RANK() OVER (ORDER BY AVG(FraudFound_P) DESC) AS fraud_rank
    FROM claims GROUP BY Make HAVING COUNT(*) >= 50
)
SELECT * FROM make_stats ORDER BY fraud_rank;
