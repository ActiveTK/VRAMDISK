use std::collections::HashMap;
use std::sync::{Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_JOBS: usize = 100_000;
/// When the registry is full, up to this many of the oldest *terminal* jobs
/// are evicted to make room for a new submission, so a long-lived mount can
/// keep accepting jobs instead of hitting a permanent `TooManyJobs` wall.
const EVICT_BATCH: usize = 1_000;
const MAX_DESCRIPTOR_BYTES: usize = 1024 * 1024;
pub const JOB_CANCELLED_MESSAGE: &str = "cancelled by user";

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

/// How far along a running job is, in bytes.
///
/// Jobs here routinely move tens of gigabytes, and an elapsed-seconds counter
/// tells the user nothing about whether to keep waiting. Executors publish
/// this as they go so `status.json` can drive a determinate progress bar.
///
/// `total_bytes` is the executor's best estimate at the time it was set and
/// may be revised upward mid-job; consumers must treat `done_bytes` as
/// possibly exceeding a stale `total_bytes` rather than assuming a ratio of
/// at most 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobProgress {
    pub done_bytes: u64,
    pub total_bytes: u64,
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
    pub progress: Option<JobProgress>,
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
    cancel_requested: bool,
    progress: Option<JobProgress>,
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
            // Evict the oldest finished jobs; refuse only when the registry is
            // genuinely full of jobs that are still pending or running.
            let mut terminal: Vec<(u128, String)> = inner
                .jobs
                .iter()
                .filter(|(_, job)| job.state.is_terminal())
                .map(|(id, job)| (job.updated_at_ms, id.clone()))
                .collect();
            terminal.sort();
            for (_, old_id) in terminal.into_iter().take(EVICT_BATCH) {
                inner.jobs.remove(&old_id);
            }
            if inner.jobs.len() >= MAX_JOBS {
                return Err(JobSubmitError::TooManyJobs);
            }
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
                cancel_requested: false,
                progress: None,
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
        if job.cancel_requested {
            job.state = JobState::Cancelled;
            job.error = Some(JOB_CANCELLED_MESSAGE.to_string());
            job.result = result_json(id, &job.state, job.error.as_deref());
            job.updated_at_ms = now_ms();
            self.changed.notify_all();
            return Err(JobSubmitError::AlreadySubmitted);
        }
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
        if job.cancel_requested {
            job.state = JobState::Cancelled;
            job.error = Some(JOB_CANCELLED_MESSAGE.to_string());
            job.result = result_json(id, &job.state, job.error.as_deref());
            job.updated_at_ms = now_ms();
            self.changed.notify_all();
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
        job.cancel_requested = true;
        if job.state != JobState::Running {
            job.state = JobState::Cancelled;
            job.error = Some(JOB_CANCELLED_MESSAGE.to_string());
            job.result = result_json(id, &job.state, job.error.as_deref());
        }
        job.updated_at_ms = now_ms();
        self.changed.notify_all();
        true
    }

    /// Publish how far a running job has got.
    ///
    /// Deliberately does NOT signal `changed`: progress ticks arrive once per
    /// staging pass, and waking every `wait` caller for each one would be pure
    /// overhead — waiters only care about the terminal transition, which still
    /// notifies. A tick on a job that has already finished (a late update from
    /// an executor unwinding) is ignored so a terminal snapshot cannot be
    /// dragged backwards.
    pub fn set_progress(&self, id: &str, done_bytes: u64, total_bytes: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = inner.jobs.get_mut(id) else {
            return;
        };
        if job.state.is_terminal() {
            return;
        }
        job.progress = Some(JobProgress {
            done_bytes,
            total_bytes,
        });
        job.updated_at_ms = now_ms();
    }

    /// Advance a job's completed-byte count by `delta`, keeping the total it
    /// was last given. Convenient for executors that stream work in passes and
    /// only know the increment they just finished.
    pub fn advance_progress(&self, id: &str, delta: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = inner.jobs.get_mut(id) else {
            return;
        };
        if job.state.is_terminal() {
            return;
        }
        let current = job.progress.unwrap_or(JobProgress {
            done_bytes: 0,
            total_bytes: 0,
        });
        job.progress = Some(JobProgress {
            done_bytes: current.done_bytes.saturating_add(delta),
            total_bytes: current.total_bytes,
        });
        job.updated_at_ms = now_ms();
    }

    pub fn progress(&self, id: &str) -> Option<JobProgress> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .jobs
            .get(id)
            .and_then(|job| job.progress)
    }

    pub fn cancel_requested(&self, id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .jobs
            .get(id)
            .map(|job| job.cancel_requested)
            .unwrap_or(false)
    }

    pub fn finish_cancelled(&self, id: &str) {
        self.finish(
            id,
            JobState::Cancelled,
            result_json(id, &JobState::Cancelled, Some(JOB_CANCELLED_MESSAGE)),
            Some(JOB_CANCELLED_MESSAGE.to_string()),
        );
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
            // A job that succeeded processed everything it set out to, so snap
            // the counter to the total: without this the last partial pass
            // leaves a terminal status.json reporting e.g. 97%, which reads as
            // "it stopped early" rather than "it finished".
            if state == JobState::Succeeded {
                if let Some(progress) = job.progress.as_mut() {
                    progress.done_bytes = progress.total_bytes;
                }
            }
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
            "  \"progress\": {},\r\n",
            "  \"error\": {}\r\n",
            "}}\r\n"
        ),
        json_escape(&job.id),
        job.state.as_str(),
        job.state.is_terminal(),
        job.submitted_at_ms,
        job.updated_at_ms,
        progress_json(job.progress),
        json_string_or_null(job.error.as_deref()),
    )
}

/// Render a job's byte counters, or `null` for an executor that has not
/// reported any. Consumers must handle the `null` case: not every job kind
/// can know its total up front.
fn progress_json(progress: Option<JobProgress>) -> String {
    match progress {
        Some(p) => format!(
            "{{\"done_bytes\": {}, \"total_bytes\": {}}}",
            p.done_bytes, p.total_bytes
        ),
        None => "null".to_string(),
    }
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
        progress: job.progress,
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

    #[test]
    fn cancelling_running_job_sets_request_until_worker_finishes() {
        let jobs = JobRegistry::default();
        jobs.reserve("job3").unwrap();
        jobs.complete_submission("job3", br#"{"op":"hash"}"#)
            .unwrap();
        assert!(jobs.start("job3").is_some());

        assert!(jobs.cancel("job3"));
        let running = jobs.snapshot("job3").unwrap();
        assert_eq!(running.state, JobState::Running);
        assert!(jobs.cancel_requested("job3"));

        jobs.finish_cancelled("job3");
        let snap = jobs.wait("job3").unwrap();
        assert_eq!(snap.state, JobState::Cancelled);
        assert!(snap.result.contains(JOB_CANCELLED_MESSAGE));
    }

    #[test]
    fn progress_is_null_until_an_executor_reports() {
        let jobs = JobRegistry::default();
        jobs.reserve("job4").unwrap();
        jobs.complete_submission("job4", br#"{"op":"encode"}"#)
            .unwrap();
        assert!(jobs.start("job4").is_some());

        let snap = jobs.snapshot("job4").unwrap();
        assert_eq!(snap.progress, None);
        assert!(status_json(&snap).contains("\"progress\": null"));

        jobs.set_progress("job4", 0, 4096);
        jobs.advance_progress("job4", 1024);
        let snap = jobs.snapshot("job4").unwrap();
        assert_eq!(
            snap.progress,
            Some(JobProgress {
                done_bytes: 1024,
                total_bytes: 4096,
            })
        );
        assert!(status_json(&snap)
            .contains("\"progress\": {\"done_bytes\": 1024, \"total_bytes\": 4096}"));
    }

    #[test]
    fn succeeding_snaps_progress_to_the_total() {
        // A final partial pass must not leave a finished job reporting 97%.
        let jobs = JobRegistry::default();
        jobs.reserve("job5").unwrap();
        jobs.complete_submission("job5", br#"{"op":"hash"}"#)
            .unwrap();
        assert!(jobs.start("job5").is_some());
        jobs.set_progress("job5", 900, 1000);

        jobs.succeed("job5", result_json("job5", &JobState::Succeeded, None));
        let snap = jobs.snapshot("job5").unwrap();
        assert_eq!(snap.progress.unwrap().done_bytes, 1000);
    }

    #[test]
    fn progress_updates_after_a_job_is_terminal_are_ignored() {
        // An executor unwinding after cancellation must not drag a terminal
        // job's counters backwards.
        let jobs = JobRegistry::default();
        jobs.reserve("job6").unwrap();
        jobs.complete_submission("job6", br#"{"op":"hash"}"#)
            .unwrap();
        assert!(jobs.start("job6").is_some());
        jobs.set_progress("job6", 500, 1000);
        jobs.finish_cancelled("job6");

        jobs.set_progress("job6", 0, 0);
        jobs.advance_progress("job6", 123);
        assert_eq!(jobs.progress("job6").unwrap().done_bytes, 500);
    }
}
