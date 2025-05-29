ALTER TABLE model_metadata
    ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN file_type VARCHAR(255);

ALTER TABLE power_plant_v2
    ADD COLUMN longitude NUMERIC,
    ADD COLUMN latitude NUMERIC;