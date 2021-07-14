## Dynamic Test

### Files

- **dynamic_test.py**
  - It generates a few requests in the csv file with "Job_id" and "Length". Then it get the request latency and add the "Latency" column to the csv file. Run successfully.
- **async_dynamic_test.py**
  - "dynamic_test.py" with asyncio. Fails.

- **compare**/
  - The "test.py" and "async_test.py" under it are the simplified case of reproduction of the error. "test.py" can run successfully but "async_test.py", which adds asyncio, fails.

- **stop.py**
  - It stops all the containers in the Clipper cluster.

- **utils.py**
  - Functions for utilization.