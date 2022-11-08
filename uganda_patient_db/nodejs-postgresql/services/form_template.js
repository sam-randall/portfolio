const db = require('./db');
const helper = require('../helper');
const config = require('../config');


const FORM_METADATA_TABLE_NAME = `form_md`




function validateCreate(patient) {
  let messages = [];

  console.log(patient);

  if (!patient) {
    messages.push('No object is provided');
  }

  if (!patient.nin) {
    messages.push('Nin is empty or nil');
  }

  if (!patient.name) {
    messages.push('Name is empty or nil');
  }

  if (patient.phone && patient.phone.length > 15) {
    messages.push('Phone cannot be longer than 255 characters');
  }

  if (patient.address && patient.address.length > 255) {
    messages.push('Address cannot be longer than 255 characters');
  }

  if (patient.ward && patient.ward.length > 255) {
    messages.push('Ward cannot be longer than 255 characters');
  }

  if (patient.name && patient.name.length > 255) {
    messages.push('Patient Name cannot be longer than 255 characters');
  }

  // TODO Implement validation for integer types: sex, nin, 


  if (messages.length) {
    let error = new Error(messages.join());
    error.statusCode = 400;

    throw error;
  }
}

//  CREATE TABLE table_name ....
// INSERT INTO table_name 
 // data

// SELECT COLUMNS FROM table_name : get method. 
// WHERE condition on rows of table.

const fetchMaximumID = async () => {
	const query = `SELECT MAX("form_id") AS "max_id" FROM form_md;`;
    // await client.connect();   // creates connection
    try {
        console.log('attempting...')
        const a =  await db.query(query);  // sends query
        if (a.length == 1) {
          return a[0].max_id
        } else if (a.length == 0) {
          return 1
        } else {
          return -1
        }
        
    } finally {
        // await client.end();   // closes connection
    }
};
`CREATE TABLE table_name
  (fields[0], fields[1], ...)
`
const getIDForName = async (form_name) =>{
	const query = `SELECT * FROM form_md WHERE form_name = '${form_name}';`;
    // await client.connect();   // creates connection
    try {
        console.log('attempting...', query)
        const a =  await db.query(query);  // sends query
        
        if (a.length == 1) {
          return a[0].form_id
        } else if (a.length == 0) {
          return 1
        } else {
          return -1
        }
        
    } finally {
        // await client.end();   // closes connection
    }
};

async function add_form_meta_data(created_by, form_name) {


  const max_id = await fetchMaximumID();
  if (max_id > -1) {

    console.log(max_id);
    let fields = `form_id, form_name, creator, created_at`

    let form_id = max_id + 1;
    let now = new Date().toISOString()
  
    console.log(now);
    
    let values = `${form_id},'${form_name}',${created_by},TO_TIMESTAMP('${now}', 'YYYY-MM-DD"T"HH24:MI:SSFF3"Z"')`
    var query = `INSERT INTO ${FORM_METADATA_TABLE_NAME} (${fields}) VALUES (${values}) RETURNING *`
    console.log(query)
    const result = await db.query(query);
    console.log(`Inserting meta data for ${form_name} ${result}`)

  }
  // console.log(`Got max ID: ${max_id.value}`)
  // console.log('result', result.toString()))
}

async function add_form_for_patient(patient_id, form_name, form_data) {

   

    console.log("Adding form" , patient_id ,form_name , form_data)

    //'INSERT INTO cervix_requests (nin, date, via_result, needs_biopsy, map, notes, next_review_date, screening_staff_name) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *',
    // [request.nin, request.date, request.via_result, request.needs_biopsy, request.map, request.notes, request.next_review_date, request.screening_staff_name]
    var array_fields = [`"nin"`]
    var array_data = [`'${patient_id}'`]

    for (const [field_name, field_val] of Object.entries(form_data)) {  
      array_fields.push(`"${field_name}"`)
      var val = null;

      // TODO: Change this 
      if (typeof(field_val) == "string") {
        val = `'` + field_val + `'`
      } else {
        val = field_val
      }

      array_data.push(val)
    }


    var fields_list_as_string = `(` + array_fields.join(",") + `)`
    var fields_data_as_string = `(` + array_data.join(", ") + `)`

    var query = `INSERT INTO "${form_name}" ` 
      + fields_list_as_string + ` VALUES ` + fields_data_as_string ;
    console.log("Line 142 Query", query)

    const result = await db.query(query);
    console.log("Success.", query)
    let message = 'Error in creating quote';
  
    if (result.length) {
      message = `${form_name} for ${patient_id} request created successfully`;
    }


    const form_id = await getIDForName(form_name);

    console.log("Got id", form_id)
    let fields = [`nin`, `form_id`, `form_name`];
    let field_values = [patient_id, form_id, `'${form_name}'`];

    var connections_query = `INSERT INTO forms (${fields.join(", ")}) VALUES (${field_values.join(", ")}) RETURNING *;`
    console.log("Attempting...", connections_query)
    const result_connections_query = await db.query(connections_query);
    console.log("Success adding...")
    return {message};
}

async function create(form) {
    // form is a dictionary of 
    // {
    //   "name" : "my_name",
    //   "fields" : {
    //     "name": "VARCHAR(256)",
    //     "age": "SMALLINT",
    //     "is_alive": "BOOL"
    //   }

    // }
    let meta_add_result = await add_form_meta_data(form.created_by, form.name);
    var errors = 0;
    console.log("Creating form...", form)
    console.log("Dropping Table if exists...", form.name)

    let query = 'DROP TABLE IF EXISTS ' + form.name

    const result = await db.query(query);
    console.log("Dropped Table", result)

    let length = Object.keys(form.fields).length

    // build fields with data types.
    const allowable_types = ["smallint", "bool", "date", "varchar(255)"]
    
    if (length > 0) {
      var arrayOfQueries = [`(nin INT NOT NULL REFERENCES patients (nin),`]
      // arrayOfQueries[0] = string
      var counter = 0
      for (const [p, val] of Object.entries(form.fields)) {
        // p is field name, val is field type.
        
        if (p == "nin") {
          errors += 1
          continue // do not over write.
          
        } else if  (!allowable_types.includes(val)) { // check to make sure ok.
          errors += 1
          continue
          
        }

        var item = `${p} ${val}`
        
        counter = counter + 1
        if (counter != Object.keys(form.fields).length) {
          console.log("counter", counter, form.fields.length)
          item += `,`
        }
        arrayOfQueries.push(item)

        console.log("adding", item)
        console.log("arr", arrayOfQueries)
      }
      arrayOfQueries.push(`)`)
      console.log("querying string", arrayOfQueries)
      const fields = arrayOfQueries.join(" ")

      console.log("fields", fields)
      // TODO: Update this to take in all fields and their data type.
      let create_table_query = 'CREATE TABLE ' + form.name +  ' ' + fields
      console.log("Querying", create_table_query)
      const create_result = await db.query(create_table_query);
      console.log("Creation", create_result)
    } else {
      errors += 1


    }




    // TODO: add error checking.

    // TODO CREATE
    // TO
//   const result = await db.query(
//     'INSERT INTO patients_v2 (nin, patient_name, outpatient_id_number, sex, ward, phone_number, address, birth_date) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *',
//     [patient.nin, patient.name, patient.id, patient.sex, patient.ward, patient.phone_number, patient.address, patient.birth_date]
//   );
  let message = 'Success';

//   if (result.length) {
//     message = 'Quote created successfully';
//   }

//   return {message};
    return { message };
}


module.exports = {
  add_form_for_patient,
  create
}