curl -i -X POST -H 'Accept: application/json' \
-H 'Content-type: application/json' http://localhost:3000/form/add \
 --data '{"nin": 1,"name":"sam_lab_result_001","fields":{"notes":"lab result notes","severity":9}}'
