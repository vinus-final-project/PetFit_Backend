-- PetFit 초기 스키마 (MySQL 8.0)
--
-- DB 명세서 · DB 설계서를 기준으로 생성한 DDL이다.
-- SQLAlchemy 모델(app/models/)에서 자동 생성했으므로 직접 수정하지 않는다.

BEGIN;

CREATE TABLE analysis (
	analysis_id BIGINT NOT NULL AUTO_INCREMENT, 
	device_id VARCHAR(36) NOT NULL, 
	animal_group VARCHAR(30) NOT NULL, 
	space_type VARCHAR(30) NOT NULL, 
	status VARCHAR(20) NOT NULL DEFAULT 'PENDING', 
	stage VARCHAR(30), 
	progress INTEGER NOT NULL DEFAULT '0', 
	error_message TEXT, 
	retry_count INTEGER NOT NULL DEFAULT '0', 
	video_path VARCHAR(255) NOT NULL, 
	capture_duration NUMERIC(4, 1) NOT NULL DEFAULT '0', 
	frame_count INTEGER NOT NULL DEFAULT '0', 
	thumbnail_path VARCHAR(255), 
	occupancy_ratio NUMERIC(5, 4) NOT NULL DEFAULT '0', 
	total_score INTEGER NOT NULL DEFAULT '0', 
	safety_score INTEGER NOT NULL DEFAULT '0', 
	activity_score INTEGER NOT NULL DEFAULT '0', 
	rest_score INTEGER NOT NULL DEFAULT '0', 
	environment_score INTEGER NOT NULL DEFAULT '0', 
	risk_factors JSON NOT NULL DEFAULT (JSON_ARRAY()), 
	analysis_result JSON NOT NULL DEFAULT (JSON_ARRAY()), 
	created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6), 
	completed_at DATETIME(6), 
	PRIMARY KEY (analysis_id), 
	CONSTRAINT ck_analysis_status CHECK (status IN ('PENDING','PROCESSING','COMPLETED','FAILED')), 
	CONSTRAINT ck_analysis_stage CHECK (stage IS NULL OR stage IN ('FRAME_EXTRACTION','OBJECT_DETECTION','OBJECT_TRACKING','FRAME_SELECTION','RISK_MARKING','SCORE_CALCULATION','ENVIRONMENT_ANALYSIS')), 
	CONSTRAINT ck_analysis_animal_group CHECK (animal_group IN ('small_dog','large_dog','cat')), 
	CONSTRAINT ck_analysis_space_type CHECK (space_type IN ('living_room','bedroom','kitchen','balcony')), 
	CONSTRAINT ck_analysis_progress CHECK (progress BETWEEN 0 AND 100), 
	CONSTRAINT ck_analysis_retry_count CHECK (retry_count BETWEEN 0 AND 3), 
	CONSTRAINT ck_analysis_total_score CHECK (total_score BETWEEN 0 AND 100), 
	CONSTRAINT ck_analysis_safety_score CHECK (safety_score BETWEEN 0 AND 100), 
	CONSTRAINT ck_analysis_activity_score CHECK (activity_score BETWEEN 0 AND 100), 
	CONSTRAINT ck_analysis_rest_score CHECK (rest_score BETWEEN 0 AND 100), 
	CONSTRAINT ck_analysis_environment_score CHECK (environment_score BETWEEN 0 AND 100), 
	CONSTRAINT ck_analysis_occupancy_ratio CHECK (occupancy_ratio BETWEEN 0 AND 1), 
	CONSTRAINT ck_analysis_capture_duration CHECK (capture_duration = 0 OR capture_duration BETWEEN 3 AND 30), 
	CONSTRAINT ck_analysis_frame_count CHECK (frame_count = 0 OR frame_count BETWEEN 15 AND 30), 
	CONSTRAINT ck_analysis_stage_consistency CHECK ((status = 'PROCESSING' AND stage IS NOT NULL) OR (status = 'FAILED') OR (status IN ('PENDING','COMPLETED') AND stage IS NULL)), 
	CONSTRAINT ck_analysis_progress_consistency CHECK ((status = 'PENDING' AND progress = 0) OR (status = 'COMPLETED' AND progress = 100) OR (status IN ('PROCESSING','FAILED'))), 
	CONSTRAINT ck_analysis_completed_at CHECK ((status = 'COMPLETED') = (completed_at IS NOT NULL)), 
	CONSTRAINT ck_analysis_error_message CHECK ((status = 'FAILED') = (error_message IS NOT NULL))
)CHARSET=utf8mb4 ENGINE=InnoDB COLLATE utf8mb4_unicode_ci;

CREATE INDEX ix_analysis_device_created ON analysis (device_id, created_at DESC);

CREATE INDEX ix_analysis_animal_group ON analysis (animal_group);

CREATE INDEX ix_analysis_device_status_created ON analysis (device_id, status, created_at DESC);

CREATE TABLE detected_object (
	object_id BIGINT NOT NULL AUTO_INCREMENT, 
	analysis_id BIGINT NOT NULL, 
	object_name VARCHAR(50) NOT NULL, 
	confidence NUMERIC(5, 4) NOT NULL, 
	detection_frame_count INTEGER NOT NULL, 
	risk_level VARCHAR(20) NOT NULL DEFAULT 'SAFE', 
	frame_number INTEGER NOT NULL, 
	marked_image_path VARCHAR(255), 
	x NUMERIC(5, 4) NOT NULL, 
	y NUMERIC(5, 4) NOT NULL, 
	width NUMERIC(5, 4) NOT NULL, 
	height NUMERIC(5, 4) NOT NULL, 
	PRIMARY KEY (object_id), 
	CONSTRAINT ck_detected_object_risk_level CHECK (risk_level IN ('HIGH','MEDIUM','LOW','SAFE')), 
	CONSTRAINT ck_detected_object_confidence CHECK (confidence BETWEEN 0 AND 1), 
	CONSTRAINT ck_detected_object_frame_count CHECK (detection_frame_count >= 1), 
	CONSTRAINT ck_detected_object_frame_number CHECK (frame_number >= 1), 
	CONSTRAINT ck_detected_object_xy CHECK (x BETWEEN 0 AND 1 AND y BETWEEN 0 AND 1), 
	CONSTRAINT ck_detected_object_wh CHECK (width > 0 AND width <= 1 AND height > 0 AND height <= 1), 
	CONSTRAINT ck_detected_object_bounds CHECK (x + width <= 1 AND y + height <= 1), 
	CONSTRAINT ck_detected_object_marking CHECK (risk_level <> 'SAFE' OR marked_image_path IS NULL), 
	FOREIGN KEY(analysis_id) REFERENCES analysis (analysis_id) ON DELETE CASCADE
)CHARSET=utf8mb4 ENGINE=InnoDB COLLATE utf8mb4_unicode_ci;

CREATE INDEX ix_detected_object_analysis ON detected_object (analysis_id);

CREATE INDEX ix_detected_object_risk ON detected_object (risk_level);

CREATE INDEX ix_detected_object_name ON detected_object (object_name);

CREATE TABLE recommendation (
	recommendation_id BIGINT NOT NULL AUTO_INCREMENT, 
	analysis_id BIGINT NOT NULL, 
	recommendation_type VARCHAR(30) NOT NULL, 
	recommendation_text TEXT NOT NULL, 
	priority INTEGER NOT NULL DEFAULT '1', 
	source VARCHAR(20) NOT NULL DEFAULT 'DETECTED', 
	PRIMARY KEY (recommendation_id), 
	CONSTRAINT ck_recommendation_type CHECK (recommendation_type IN ('SAFETY','ACTIVITY','REST','ENVIRONMENT')), 
	CONSTRAINT ck_recommendation_source CHECK (source IN ('DETECTED','OBSERVED')), 
	CONSTRAINT ck_recommendation_priority CHECK (priority >= 1), 
	CONSTRAINT uq_recommendation_priority UNIQUE (analysis_id, priority), 
	FOREIGN KEY(analysis_id) REFERENCES analysis (analysis_id) ON DELETE CASCADE
)CHARSET=utf8mb4 ENGINE=InnoDB COLLATE utf8mb4_unicode_ci;

CREATE INDEX ix_recommendation_analysis ON recommendation (analysis_id);

COMMIT;
