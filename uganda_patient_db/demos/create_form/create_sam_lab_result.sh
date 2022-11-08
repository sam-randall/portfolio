curl -i -X POST -H 'Accept: application/json' \
-H 'Content-type: application/json' http://localhost:3000/form/create \
 --data '{"nin": 1,"created_by": 1997,"name":"sam_lab_result_001","fields":{"notes":"varchar(255)","severity":"smallint"}}'
