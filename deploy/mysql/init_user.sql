-- PetFit 계정·스키마 생성
--
--   mysql -u root -p < deploy/mysql/init_user.sql
--
-- 실행 전에 아래 두 비밀번호를 바꾼다. 그대로 두지 않는다.
-- 앱 비밀번호는 .env 의 DB_PASSWORD 와, 팀원 비밀번호는 deploy/TEAM.md 안내와 일치해야 한다.
--
-- 스키마를 여기서 미리 만든다. scripts/init_db.py 도 없으면 만들지만,
-- 그러려면 petfit 계정에 전역 CREATE 권한이 필요하다. 스키마를 먼저 만들어
-- 두면 init_db.py 는 "이미 존재"로 넘어가고 계정 권한을 petfit.* 로 좁힐 수 있다.

CREATE DATABASE IF NOT EXISTS `petfit`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 앱 계정. 백엔드가 쓴다.
-- ---------------------------------------------------------------------
-- 'localhost' 와 '127.0.0.1' 을 모두 만든다. MySQL 은 유닉스 소켓 접속을
-- 'localhost' 로, TCP 루프백 접속을 '127.0.0.1' 로 구분해 매칭한다.
-- 드라이버가 어느 쪽을 고르든 통과시키려면 둘 다 있어야 한다.

CREATE USER IF NOT EXISTS 'petfit'@'localhost' IDENTIFIED BY 'CHANGE_ME_APP';
CREATE USER IF NOT EXISTS 'petfit'@'127.0.0.1' IDENTIFIED BY 'CHANGE_ME_APP';

-- init_db.py 가 CREATE TABLE · DROP TABLE(--drop) 을 수행하므로 DDL 이 필요하다.
GRANT ALL PRIVILEGES ON `petfit`.* TO 'petfit'@'localhost';
GRANT ALL PRIVILEGES ON `petfit`.* TO 'petfit'@'127.0.0.1';

-- ---------------------------------------------------------------------
-- 팀원 계정. Cloudflare Tunnel 을 거쳐 들어온다.
-- ---------------------------------------------------------------------
-- 터널 트래픽은 이 장비의 cloudflared 가 127.0.0.1:3306 으로 중계하므로,
-- MySQL 에는 접속 원본이 127.0.0.1 로 보인다. 팀원 IP 로는 제한할 수 없다.
-- 접근 통제는 Cloudflare Access 정책이 담당한다(deploy/README.md 참고).
--
-- 기본은 조회 전용이다. 계정이 공유되는 이상 실수 한 번의 범위를 좁혀 둔다.
-- 데이터를 넣고 지워야 하는 사람이 생기면 아래 주석을 풀어 개별 계정을 만든다.

CREATE USER IF NOT EXISTS 'petfit_team'@'127.0.0.1' IDENTIFIED BY 'CHANGE_ME_TEAM';
GRANT SELECT, SHOW VIEW ON `petfit`.* TO 'petfit_team'@'127.0.0.1';

-- 쓰기가 필요한 팀원용. 이름을 사람마다 다르게 만든다. 누가 지웠는지 남는다.
-- CREATE USER IF NOT EXISTS 'petfit_dev_minji'@'127.0.0.1' IDENTIFIED BY 'CHANGE_ME';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON `petfit`.* TO 'petfit_dev_minji'@'127.0.0.1';

FLUSH PRIVILEGES;

-- 확인
SELECT user, host, plugin FROM mysql.user WHERE user LIKE 'petfit%';
SHOW GRANTS FOR 'petfit'@'127.0.0.1';
SELECT @@global.time_zone AS global_tz, @@session.time_zone AS session_tz;
SELECT @@character_set_server AS charset, @@collation_server AS collation;
SELECT @@bind_address AS bind_address;
