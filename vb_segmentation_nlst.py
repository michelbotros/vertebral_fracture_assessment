import gcapi
import time

from tqdm import tqdm
from pathlib import Path
from typing import Iterable


class ThrottledRetries:  # implements an exponential backoff strategy
    def __init__(self, attempts: int, interval: int):
        self.attempts = attempts
        self.interval = interval

    def __iter__(self) -> Iterable[int]:
        for n in range(self.attempts):
            if n > 0:
                time.sleep(min(self.interval * 1.5 ** n, 300))
            yield n


class GrandChallengeAlgorithm:
    def __init__(self, client: gcapi.Client):
        self.client = client
        self.algorithm = 'vertebral-body-segmentation'
        self.retries = ThrottledRetries(attempts=25, interval=15)

        # Query API to get average runtime
        algorithm_details = self.client.get_algorithm(self.algorithm)
        average_duration = algorithm_details["average_duration"]
        if average_duration:
            self.headstart = int(average_duration * 0.75)
        else:
            self.headstart = 30  # default to 30 seconds

        # Query API to get a list of existing results
        self.existing_outputs = dict()
        for job in self.client.algorithm_jobs.iterate_all({'algorithm_image__algorithm': algorithm_details['pk']}):
            if job['status'] != 'Succeeded':
                continue
            for output in job['outputs']:
                if output['interface']['slug'] == 'vertebral-body':
                    break
            else:
                continue
            self.existing_outputs[job['inputs'][0]['image']] = job['outputs']

    def run(self, image_url, output_file: Path):
        """Uploads the image to grand challenge and downloads the resulting segmentation mask"""
        # Check if algorithm ran already
        if image_url in self.existing_outputs:
            tqdm.write('Output exists already, downloading...')
            self._download_results(self.existing_outputs[image_url], output_file)
            return

        # Start job on GC
        job = self.client.run_external_job(
            algorithm=self.algorithm,
            inputs={"ct-image": image_url}
        )

        # Wait for job to complete
        time.sleep(self.headstart)
        for _ in self.retries:
            try:
                job = self.client.algorithm_jobs.detail(job["pk"])
            except (IOError, IndexError):
                continue

            status = job["status"]
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f'Algorithm "{self.algorithm}" failed')
            elif status == "Succeeded":
                self._download_results(job["outputs"], output_file)
                break
        else:
            raise TimeoutError

    def _download_results(self, outputs: Iterable, output_file: Path):
        for output in outputs:
            if output["interface"]["slug"] != "vertebral-body":
                continue

            for _ in self.retries:
                try:
                    image_details = self.client(url=output["image"])
                except IOError:
                    continue
                else:
                    break
            else:
                raise TimeoutError

            for file in image_details["files"]:
                # Skip images that are not mha files
                if file["image_type"] != "MHD":
                    continue

                # Download data and dump into file
                if output_file.suffix != ".mha":
                    raise ValueError("Output file needs to have .mha extension")

                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("wb") as fp:
                    fp.write(self.client(url=file["file"]).content)

                return  # there is only one mask that we need to save


def main():
    client = gcapi.Client()
    dstdir = Path(r'D:/Temp/nlst_vertebral_bodies')

    # Query reader study to get a list of scans that were annotated by Matthieu
    print('Looking up reader study UUID')
    for rs in client.reader_studies.iterate_all({'slug': 'vertebral-assessment'}):
        reader_study_uuid = rs['pk']
        print(f'> {reader_study_uuid}')
        break
    else:
        raise RuntimeError('Could not find reader study')

    image_urls = set()
    for answer in tqdm(
        client.reader_studies.answers.iterate_all({'question__reader_study': reader_study_uuid}),
        desc='Retrieving list of annotated images'
    ):
        if answer['creator'] == 'm_rutten':
            image_urls.update(answer['images'])

    # Run vertebral body segmentation for all images
    algorithm = GrandChallengeAlgorithm(client)
    for url in tqdm(image_urls, desc='Running segmentation algorithm'):
        image = client(url=url)
        name = image['name'][:-4]
        output_file = dstdir / f'{name}.mha'

        if 'mask' in name or output_file.exists():
            continue

        algorithm.run(url, output_file)


if __name__ == "__main__":
    main()
