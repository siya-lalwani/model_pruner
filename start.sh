python manage.py migrate
python manage.py collectstatic --noinput
gunicorn prunEd.wsgi:application
