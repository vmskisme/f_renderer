use crossbeam::channel::{self, Receiver, Sender};
use std::sync::mpsc::channel;
use std::sync::{mpsc::SendError, Arc, Mutex};
use std::thread::{self, JoinHandle};


pub trait FnBox {
    fn call_box(self: Box<Self>);
}

impl<F: FnOnce()> FnBox for F {
    fn call_box(self: Box<F>) {
        (*self)()
    }
}

type Task = Box<dyn FnBox + Send>;

pub struct ThreadPool {
    max_threads: u32,
    handles: Vec<JoinHandle<()>>,
    tx: Sender<Task>,
    rx: Receiver<Task>,
    task_num: u32,
    p_tx: Sender<()>,
    p_rx: Receiver<()>
}

impl ThreadPool {
    pub fn new(max_threads: u32) -> Self {
        let (tx, rx) = channel::unbounded::<Task>();
        let mut handles = vec![];

        let (s, r) = channel::unbounded::<()>();
        for _ in 0..max_threads {
            let rx = rx.clone();
            let s = s.clone();
            handles.push(thread::spawn(move || {
                while let Ok(task) = rx.recv() {
                    task.call_box();
                    s.send(()).unwrap();
                }
            }))
        }

        let mut pool = Self {
            max_threads: max_threads,
            handles: handles,
            tx: tx,
            rx: rx,
            task_num: 0,
            p_rx: r,
            p_tx: s,
        };

        pool
    }

    pub fn execute(&mut self, task: Task) {
        self.task_num += 1;
        self.tx.send(task).unwrap();
    }

    pub fn join_all(&mut self) {
        let rx = self.p_rx.clone();
        let mut finished_count = 0;
        let task_num = self.task_num;
        thread::spawn(move || {
            while let Ok(_) = rx.recv() {
                finished_count += 1;

                if finished_count >= task_num {break;}
            }
        }).join().unwrap();

        self.task_num = 0;
    }
}