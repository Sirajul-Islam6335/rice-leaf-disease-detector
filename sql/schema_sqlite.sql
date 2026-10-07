-- Schema for the detector app (SQLite dialect), generated with:
--   python manage.py sqlmigrate detector 0001
-- On PostgreSQL run the same command with DB_ENGINE=postgres to get that dialect.

BEGIN;
--
-- Create model Disease
--
CREATE TABLE "detector_disease" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "name" varchar(64) NOT NULL UNIQUE, "slug" varchar(64) NOT NULL UNIQUE, "pathogen" varchar(120) NOT NULL, "kind" varchar(16) NOT NULL, "color" varchar(7) NOT NULL, "sample_image" varchar(200) NOT NULL, "summary" text NOT NULL, "symptoms" text NOT NULL, "favoured_by" text NOT NULL, "management" text NOT NULL, "looks_like" text NOT NULL, "display_order" smallint unsigned NOT NULL CHECK ("display_order" >= 0));
--
-- Create model Prediction
--
CREATE TABLE "detector_prediction" ("id" char(32) NOT NULL PRIMARY KEY, "image" varchar(100) NOT NULL, "original_filename" varchar(255) NOT NULL, "source" varchar(10) NOT NULL, "confidence" real NOT NULL, "probabilities" text NOT NULL CHECK ((JSON_VALID("probabilities") OR "probabilities" IS NULL)), "uncertain" bool NOT NULL, "inference_ms" real NOT NULL, "model_version" varchar(80) NOT NULL, "feedback_note" varchar(500) NOT NULL, "feedback_at" datetime NULL, "created_at" datetime NOT NULL, "predicted_disease_id" bigint NOT NULL REFERENCES "detector_disease" ("id") DEFERRABLE INITIALLY DEFERRED, "true_disease_id" bigint NULL REFERENCES "detector_disease" ("id") DEFERRABLE INITIALLY DEFERRED, "user_id" integer NULL REFERENCES "auth_user" ("id") DEFERRABLE INITIALLY DEFERRED, CONSTRAINT "prediction_confidence_0_1" CHECK (("confidence" >= 0.0 AND "confidence" <= 1.0)));
CREATE INDEX "detector_prediction_created_at_736732b2" ON "detector_prediction" ("created_at");
CREATE INDEX "detector_prediction_predicted_disease_id_d4aa7437" ON "detector_prediction" ("predicted_disease_id");
CREATE INDEX "detector_prediction_true_disease_id_cad2085a" ON "detector_prediction" ("true_disease_id");
CREATE INDEX "detector_prediction_user_id_b9d8eadf" ON "detector_prediction" ("user_id");
CREATE INDEX "pred_disease_created_idx" ON "detector_prediction" ("predicted_disease_id", "created_at");
CREATE INDEX "pred_user_created_idx" ON "detector_prediction" ("user_id", "created_at");
COMMIT;
