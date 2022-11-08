

curl -i -X POST -H 'Accept: application/json' \
    -H 'Content-type: application/json' http://localhost:3000/requests/cervical \
    --data '{"nin": 3, "date": "2017-05-07T02:35:16.385Z", "via_result": "suspicious", "needs_biopsy": 0, "map": "upper left", "notes": "come back in three months", "next_review_date": "2017-09-07T02:35:16.385Z", "screening_staff_name": "Sam Randall"}'

