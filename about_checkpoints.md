# Pre-Empting and Checkpointing


## What is "pre-empting"?

Pre-Emptibility is feature of computing systems that allow your compute system to shut down with notice.  The usual trade off is that the resources are cheaper or easier to obtain.

Pre-empted resources can be found on:
* The `owners` queue of Sherlock
* Marlowe Basic
* Spot instances in Google Cloud


## Why do we care?

If your code can handle pre-emptibility, you may greatly expand the resources available to you, and the speed in which you can obtain them.

## Robust and Restartable Code

* State Management
  *   Your code should be able to detect what stages to run and what stages to skip
  *  I.e. if a file exists, skip the step to create that file 
* “Idempotency”
  * Your code should produce the same result every time you run it, no matter how many times you run it
* Checkpointing
  * Graceful restartability


