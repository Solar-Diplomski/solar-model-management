DROP TABLE IF EXISTS public.events CASCADE;
DROP TABLE IF EXISTS public.models CASCADE;
DROP TABLE IF EXISTS public.power_plant CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP TABLE IF EXISTS public.weather_forecast CASCADE;
DROP SEQUENCE IF EXISTS public.power_plant_plant_id_seq CASCADE;
DROP SEQUENCE IF EXISTS public.users_id_seq CASCADE;

ALTER TABLE power_plant_v2 RENAME TO power_plant; 