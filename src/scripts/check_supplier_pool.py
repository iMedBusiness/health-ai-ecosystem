import duckdb

con = duckdb.connect()

# IMPORTANT: use delta_scan for Delta Lake tables
df = con.execute("""
    SELECT
        supplier_id,
        supplier_name,
        price_per_unit,
        lead_time_days,
        reliability_score
    FROM delta_scan('data/lakehouse/suppliers/supplier_pool')
""").df()

print(df)

