DROP TABLE IF EXISTS public.events CASCADE;
DROP TABLE IF EXISTS public.models CASCADE;
DROP TABLE IF EXISTS public.power_plant CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP TABLE IF EXISTS public.weather_forecast CASCADE;
DROP SEQUENCE IF EXISTS public.power_plant_plant_id_seq CASCADE;
DROP SEQUENCE IF EXISTS public.users_id_seq CASCADE;

-- Create sequences
CREATE SEQUENCE public.power_plant_plant_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

-- Create tables
CREATE TABLE public.power_plant (
    plant_id integer NOT NULL DEFAULT nextval('public.power_plant_plant_id_seq'::regclass),
    plant_name character varying(255),
    capacity_mw double precision,
    num_panels integer,
    panel_height double precision,
    panel_width double precision,
    total_panel_surface double precision,
    panel_efficiency double precision,
    system_efficiency double precision,
    total_surface_and_efficiency double precision,
    power_dependence_on_temperature_related_to_25_celsius double precision,
    max_installed_capacity double precision,
    latitude character varying(255),
    longitude character varying(255),
    status boolean,
    models integer,
    utilization double precision,
    current_production double precision,
    CONSTRAINT power_plant_pkey PRIMARY KEY (plant_id)
);

CREATE TABLE public.models (
    model_id character varying(120) NOT NULL,
    model_name character varying(255),
    description text,
    plant_id integer,
    accuracy integer,
    status character varying(255),
    type character varying(255),
    best boolean,
    CONSTRAINT models_pkey PRIMARY KEY (model_id),
    CONSTRAINT models_plant_id_fkey FOREIGN KEY (plant_id) REFERENCES public.power_plant(plant_id)
);

CREATE TABLE public.events (
    id integer NOT NULL,
    model_id integer NOT NULL,
    status character varying(50) NOT NULL,
    datetime timestamp without time zone NOT NULL,
    description character varying(255),
    CONSTRAINT events_pkey PRIMARY KEY (id)
    -- Foreign key to models.model_id not added due to type mismatch (varchar vs integer)
);

CREATE TABLE public.users (
    id integer NOT NULL DEFAULT nextval('public.users_id_seq'::regclass),
    full_name character varying(255) NOT NULL,
    email character varying(255) NOT NULL,
    username character varying(100) NOT NULL,
    avatar_url character varying(255),
    role character varying(50) NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    active boolean,
    CONSTRAINT users_pkey PRIMARY KEY (id),
    CONSTRAINT users_email_key UNIQUE (email),
    CONSTRAINT users_username_key UNIQUE (username),
    CONSTRAINT users_role_check CHECK (role IN ('Admin', 'Editor'))
);

CREATE TABLE public.weather_forecast (
    tof timestamp without time zone NOT NULL,
    vt timestamp without time zone NOT NULL,
    barometer double precision,
    outtemp double precision,
    windspeed double precision,
    winddir integer,
    rain double precision,
    radiation integer,
    cloud_cover double precision,
    CONSTRAINT weather_forecast_pkey PRIMARY KEY (tof, vt)
);

-- Set sequence ownership
ALTER SEQUENCE public.power_plant_plant_id_seq OWNED BY public.power_plant.plant_id;
ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;

-- Insert initial data
INSERT INTO public.power_plant (plant_id, plant_name, capacity_mw, num_panels, panel_height, panel_width, total_panel_surface, panel_efficiency, system_efficiency, total_surface_and_efficiency, power_dependence_on_temperature_related_to_25_celsius, max_installed_capacity, latitude, longitude, status, models, utilization, current_production) VALUES
(1, 'SE Vis', 1.44, 11200, 1.685, 1, 18872, 0.2018, 77, 0.002932445, -0.01287104, 3.8083696, '43.03823574273269', '16.150850402782556', true, 2, 57, 0.82),
(2, 'SE Drava', 0.98, 1000, 1, 1, 1, 93, 91, NULL, NULL, NULL, '45.52121150403985', '18.664564092580623', false, 1, 60, 0.58),
(3, 'SE Kaštelir', 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, true, 3, 49, 0.49);

INSERT INTO public.models (model_id, model_name, description, plant_id, accuracy, status, type, best) VALUES
('1', 'Model A', 'Description of Model A', 1, 89, 'running', 'Linear Regression', NULL),
('2', 'Model B', 'Description of Model B.', 2, 92, 'ready', 'Mathematical', true),
('3', 'Model C', 'Description of Model C.', 1, 75, NULL, 'LSTM', NULL);

INSERT INTO public.events (id, model_id, status, datetime, description) VALUES
(915, 1, 'finished', '2024-11-08 07:21:38', NULL),
(610, 3, 'running', '2024-10-27 07:56:35', NULL),
(498, 1, 'finished', '2024-11-07 23:10:57', NULL),
(98, 3, 'running', '2024-11-03 19:32:54', NULL),
(136, 1, 'finished', '2024-10-20 14:23:58', NULL),
(991, 1, 'running', '2024-11-03 16:28:53', NULL),
(617, 3, 'finished', '2024-11-04 17:49:13', NULL),
(911, 1, 'error', '2024-11-02 19:29:01', 'Error due to timeout.'),
(952, 3, 'error', '2024-11-05 22:08:59', 'This is an error description'),
(996, 1, 'error', '2024-10-15 02:34:33', 'Error description'),
(998, 1, 'error', '2024-10-15 02:34:33', 'Error description'),
(997, 1, 'error', '2024-10-15 02:34:33', 'Error description');

INSERT INTO public.users (id, full_name, email, username, avatar_url, role, created_at, active) VALUES
(1, 'John Doe', 'john.doe@example.com', 'johndoe', 'https://i.pravatar.cc/?img=8', 'Admin', '2025-01-01 10:00:00', true),
(2, 'Jane Smith', 'jane.smith@example.com', 'janesmith', 'https://i.pravatar.cc/?img=1', 'Editor', '2025-01-02 14:30:00', true),
(3, 'Alice Johnson', 'alice.johnson@example.com', 'alicej', 'https://i.pravatar.cc/?img=5', 'Admin', '2025-01-03 08:15:00', false),
(4, 'Bob Brown', 'bob.brown@example.com', 'bobbyb', 'https://i.pravatar.cc/?img=12', 'Editor', '2025-01-04 11:45:00', false),
(5, 'Charlie White', 'charlie.white@example.com', 'charliew', 'https://i.pravatar.cc/?img=14', 'Admin', '2025-01-05 16:20:00', true);

INSERT INTO public.weather_forecast (tof, vt, barometer, outtemp, windspeed, winddir, rain, radiation, cloud_cover) VALUES
('2024-05-12 12:00:00', '2024-05-12 12:00:00', 1013.25, 25.5, 5.2, 180, 0, 600, 0.3),
('2024-05-12 12:00:00', '2024-05-12 13:00:00', 1012.75, 24.8, 4.8, 175, 0, 550, 0.4);

-- Set sequence values to match inserted data
SELECT pg_catalog.setval('public.power_plant_plant_id_seq', 3, true);
SELECT pg_catalog.setval('public.users_id_seq', 5, true);