use std::collections::HashMap;
use std::sync::{Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_JOBS: usize = 1_000_000;
const MAX_DESCRIPTOR_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    Receiving,
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Receiving => "receiving",
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Succeeded => "succeeded",
            JobState::Failed => "failed",
            JobState::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobState::Succeeded | JobState::Failed | JobState::Cancelled
        )
    }
}

#[derive(Debug, Clone)]
pub struct JobSnapshot {
    pub id: String,
    pub state: JobState,
    pub submitted_at_ms: u128,
    pub updated_at_ms: u128,
    pub descriptor: String,
    pub result: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobSubmitError {
    InvalidId,
    AlreadyExists,
    TooManyJobs,
    DescriptorTooLarge,
    NotFound,
    AlreadySubmitted,
}

#[derive(Default)]
pub struct JobRegistry {
    inner: Mutex<JobRegistryInner>,
    changed: Condvar,
}

#[derive(Default)]
struct JobRegistryInner {
    jobs: HashMap<String, JobRecord>,
}

struct JobRecord {
    state: JobState,
    submitted_at_ms: u128,
    updated_at_ms: u128,
    descriptor: String,
    result: String,
    error: Option<String>,
}

impl JobRegistry {
    pub fn reserve(&self, id: &str) -> Result<(), JobSubmitError> {
        if !is_valid_job_id(id) {
            return Err(JobSubmitError::InvalidId);
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.jobs.contains_key(id) {
            return Err(JobSubmitError::AlreadyExists);
        }
        if inner.jobs.len() >= MAX_JOBS {
            return Err(JobSubmitError::TooManyJobs);
        }
        let now = now_ms();
        inner.jobs.insert(
            id.to_string(),
            JobRecord {
                state: JobState::Receiving,
                submitted_at_ms: now,
                updated_at_ms: now,
                descriptor: String::new(),
                result: "{}\r\n".to_string(),
                error: None,
            },
        );
        self.changed.notify_all();
        Ok(())
    }

    pub fn complete_submission(&self, id: &str, descriptor: &[u8]) -> Result<(), JobSubmitError> {
        if descriptor.len() > MAX_DESCRIPTOR_BYTES {
            self.fail_reserved(id, "job descriptor is too large");
            return Err(JobSubmitError::DescriptorTooLarge);
        }
        let descriptor = String::from_utf8_lossy(descriptor).into_owned();
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = inner.jobs.get_mut(id) else {
            return Err(JobSubmitError::NotFound);
        };
        if job.state != JobState::Receiving {
            return Err(JobSubmitError::AlreadySubmitted);
        }

        job.descriptor = descriptor;
        job.state = JobState::Queued;
        job.updated_at_ms = now_ms();
        self.changed.notify_all();
        Ok(())
    }

    pub fn start(&self, id: &str) -> Option<String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let job = inner.jobs.get_mut(id)?;
        if job.state != JobState::Queued {
            return None;
        }
        job.state = JobState::Running;
        job.updated_at_ms = now_ms();
        self.changed.notify_all();
        Some(job.descriptor.clone())
    }

    pub fn succeed(&self, id: &str, result: String) {
        self.finish(id, JobState::Succeeded, result, None);
    }

    pub fn fail(&self, id: &str, message: impl Into<String>) {
        let message = message.into();
        self.finish(
            id,
            JobState::Failed,
            result_json(id, &JobState::Failed, Some(&message)),
            Some(message),
        );
    }

    pub fn cancel(&self, id: &str) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = inner.jobs.get_mut(id) else {
            return false;
        };
        if job.state.is_terminal() {
            return true;
        }
        job.state = JobState::Cancelled;
        job.error = Some("cancelled by user".to_string());
        job.result = result_json(id, &job.state, job.error.as_deref());
        job.updated_at_ms = now_ms();
        self.changed.notify_all();
        true
    }

    pub fn exists(&self, id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .jobs
            .contains_key(id)
    }

    pub fn snapshot(&self, id: &str) -> Option<JobSnapshot> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .jobs
            .get(id)
            .map(|j| snapshot(id, j))
    }

    pub fn wait(&self, id: &str) -> Option<JobSnapshot> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let job = inner.jobs.get(id)?;
            if job.state.is_terminal() {
                return Some(snapshot(id, job));
            }
            inner = self.changed.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }

    pub fn receiving_ids(&self) -> Vec<String> {
        self.ids_by(|state| *state == JobState::Receiving)
    }

    pub fn completed_ids(&self) -> Vec<String> {
        self.ids_by(JobState::is_terminal)
    }

    fn ids_by(&self, pred: impl Fn(&JobState) -> bool) -> Vec<String> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut ids: Vec<String> = inner
            .jobs
            .iter()
            .filter(|(_, job)| pred(&job.state))
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        ids
    }

    fn fail_reserved(&self, id: &str, message: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = inner.jobs.get_mut(id) {
            job.state = JobState::Failed;
            job.error = Some(message.to_string());
            job.result = result_json(id, &job.state, job.error.as_deref());
            job.updated_at_ms = now_ms();
            self.changed.notify_all();
        }
    }

    fn finish(&self, id: &str, state: JobState, result: String, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = inner.jobs.get_mut(id) {
            job.state = state;
            job.result = result;
            job.error = error;
            job.updated_at_ms = now_ms();
            self.changed.notify_all();
        }
    }
}

pub fn is_valid_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id != "."
        && id != ".."
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

pub fn status_json(job: &JobSnapshot) -> String {
    format!(
        concat!(
            "{{\r\n",
            "  \"id\": \"{}\",\r\n",
            "  \"state\": \"{}\",\r\n",
            "  \"terminal\": {},\r\n",
            "  \"submitted_at_ms\": {},\r\n",
            "  \"updated_at_ms\": {},\r\n",
            "  \"error\": {}\r\n",
            "}}\r\n"
        ),
        json_escape(&job.id),
        job.state.as_str(),
        job.state.is_terminal(),
        job.submitted_at_ms,
        job.updated_at_ms,
        json_string_or_null(job.error.as_deref()),
    )
}

fn snapshot(id: &str, job: &JobRecord) -> JobSnapshot {
    JobSnapshot {
        id: id.to_string(),
        state: job.state.clone(),
        submitted_at_ms: job.submitted_at_ms,
        updated_at_ms: job.updated_at_ms,
        descriptor: job.descriptor.clone(),
        result: job.result.clone(),
        error: job.error.clone(),
    }
}

fn result_json(id: &str, state: &JobState, error: Option<&str>) -> String {
    format!(
        concat!(
            "{{\r\n",
            "  \"id\": \"{}\",\r\n",
            "  \"state\": \"{}\",\r\n",
            "  \"ok\": {},\r\n",
            "  \"error\": {}\r\n",
            "}}\r\n"
        ),
        json_escape(id),
        state.as_str(),
        *state == JobState::Succeeded,
        json_string_or_null(error),
    )
}

fn json_string_or_null(s: Option<&str>) -> String {
    match s {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_client_supplied_ids() {
        assert!(is_valid_job_id("archive-001"));
        assert!(is_valid_job_id("abc.DEF_123"));
        assert!(!is_valid_job_id(""));
        assert!(!is_valid_job_id("a\\b"));
        assert!(!is_valid_job_id("a/b"));
    }

    #[test]
    fn reserves_and_completes_noop() {
        let jobs = JobRegistry::default();
        jobs.reserve("job1").unwrap();
        jobs.complete_submission("job1", br#"{"op":"noop"}"#)
            .unwrap();
        let descriptor = jobs.start("job1").unwrap();
        assert_eq!(descriptor, r#"{"op":"noop"}"#);
        jobs.succeed("job1", result_json("job1", &JobState::Succeeded, None));
        let snap = jobs.wait("job1").unwrap();
        assert_eq!(snap.state, JobState::Succeeded);
        assert!(snap.result.contains("\"ok\": true"));
    }

    #[test]
    fn fails_jobs_without_losing_id() {
        let jobs = JobRegistry::default();
        jobs.reserve("job2").unwrap();
        jobs.complete_submission("job2", br#"{"op":"archive.create"}"#)
            .unwrap();
        assert!(jobs.start("job2").is_some());
        jobs.fail("job2", "no GPU executor");
        let snap = jobs.snapshot("job2").unwrap();
        assert_eq!(snap.state, JobState::Failed);
        assert!(snap.result.contains("no GPU executor"));
    }
}
