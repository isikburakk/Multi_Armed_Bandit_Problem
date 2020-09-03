docker build -t MAB_img -f MAB.dockerfile .
docker run -d -p 8890:8888 -v <Directory of project>:/notebook MAB_img