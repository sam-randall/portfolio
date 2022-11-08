const db = require('./db');
const helper = require('../helper');
const config = require('../config');

async function getOneRecord(id) {
  console.log("calling getOneRecord")
  const rows = await db.query(
    'SELECT * FROM patients WHERE nin = $1', 
    [id]
  );

  const data = helper.emptyOrRows(rows);
  const meta = {id};

  return {
    data,
    meta
  }
}


async function getRecordFromNinAndFormName(id, form_name) {
  console.log("calling getOneRecord from name")
  const query = `SELECT * FROM ${form_name} WHERE nin = ${id}`
  console.log(query);
  const rows = await db.query(query );

  const data = helper.emptyOrRows(rows);
  const meta = {id};

  return {
    data,
    meta
  }
}


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


`CREATE TABLE table_name
  (fields[0], fields[1], ...)
`

async function create(patient) {
  validateCreate(patient)
  const result = await db.query(
    'INSERT INTO patients (nin, patient_name, outpatient_id_number, sex, ward, phone_number, address, birth_date) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *',
    [patient.nin, patient.name, patient.id, patient.sex, patient.ward, patient.phone_number, patient.address, patient.birth_date]
  );
  let message = 'Error in creating quote';

  if (result.length) {
    message = 'Quote created successfully';
  }

  return {message};
}


module.exports = {
  getOneRecord,
  create, 
  getRecordFromNinAndFormName
}