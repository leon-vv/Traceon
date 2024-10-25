
all: contact.html download.html technology.html index.html

contact.html: template-start.html contact-inner.html template-end.html
	cat $^ > $@

technology.html: template-start.html technology-inner.html template-end.html
	cat $^ > $@

index.html: template-start.html index-inner.html template-end.html
	cat $^ > $@

download.html: template-start.html download-inner.html template-end.html
	cat $^ > $@
