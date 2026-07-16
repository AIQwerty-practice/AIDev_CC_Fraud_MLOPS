FROM nginx:1.27-alpine

ENV NGINX_ENVSUBST_OUTPUT_DIR=/usr/share/nginx/html
COPY index.html /etc/nginx/templates/index.html.template
COPY image/ /usr/share/nginx/html/image/
