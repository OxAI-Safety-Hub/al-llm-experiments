Using Docker
============

A docker file is available which allows for iterative development and running experiments. To build a new image and use it, follow the proceeding steps.

1. Create a Weights and Biases personal access token

2. Create GitHub personal access token. Ideally use a fine-grained one which has access only to the contents of this repository.

3. Create a file named `.env` with the following contents

```bash
WANDB_KEY=
GITHUB_USER=
GITHUB_PAT=
GIT_NAME=""
GIT_EMAIL=""
```

4. Fill in the details with the W&B PAT, your GitHub username, your GitHub PAT, your name as you'd like it to appear in git commit messages, and the email you'd like to use for git commits.

5. Build the image using the following command:

```
docker build -t DOCKER_REPO:DOCKER_TAG --build-arg user=USER --secret id=my_env,src=.env .
```

replacing `DOCKER_REPO` and `DOCKER_TAG` with the appropriate details and `USER` with your desired username in the image.

6. Push the image to the Docker Hub, ready for use.