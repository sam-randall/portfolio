const { Pool, Client } = require('pg');
const config = require("./config");
const helper = require("./helper");

const PATIENT_TABLE_NAME = `patients`;


const client = new Client({
    user: config.db.user,
    host: config.db.host,
    password: config.db.password,
    port: config.db.port
})

async function create_form_metadata_table() {
    console.log(`Creating form meta data table.`)
    await client.query(`CREATE TABLE form_md ( \
        form_id INT NOT NULL, \
        form_name VARCHAR(255) NOT NULL, \
        creator INT, \
        created_at DATE
    )`)
    console.log(`Success!`)
}

async function drop_form_metadata_table() {
    console.log(`Dropping form meta data table.`)
    await client.query(`DROP TABLE IF EXISTS form_md;`)
    console.log(`Success!`)
}

async function drop_patient_table_if_necessary() {
    console.log(`Attempting to drop patient table.`)
    await client.query(`DROP TABLE IF EXISTS ${PATIENT_TABLE_NAME}`)
    console.log("Dropped patients table")
}

async function drop_patient_constraint() {
    let command = `ALTER TABLE ${PATIENT_TABLE_NAME} DROP CONSTRAINT patients_key`;
    await client.query(command);
}

async function drop_forms_table() {

    console.log(`Attempting to remove forms table.`)
    await client.query(`DROP TABLE IF EXISTS ${helper.CONNECTIONS_TABLE_NAME}`)
    console.log(`Removed forms table successfully.`)
}
async function create_forms_table() {
    console.log(`Attempting to create forms table.`)
    
    await client.query(`CREATE TABLE ${helper.CONNECTIONS_TABLE_NAME} ( \
        nin INT NOT NULL, \
        form_id VARCHAR(255) NOT NULL, \
        form_name VARCHAR(255) NOT NULL \  
    )`)

    console.log(`Created patient table successfully.`)
}


async function get_all_user_created_forms() {
    console.log(`Getting rows...`)
    let data = await client.query(`SELECT form_name FROM form_md;`)
    let rows = data.rows;
    return rows;
}

// Create architecture for patient table.
async function create_patient_table() {
    console.log(`Attempting to create_patient_table.`)
    
    await client.query(`CREATE TABLE ${PATIENT_TABLE_NAME} ( \
        nin INT NOT NULL, \
        patient_name VARCHAR(255) NOT NULL, \
        outpatient_id_number INT, \
        birth_date DATE, \
        sex SMALLINT, \
        ward VARCHAR(255),\
        phone_number VARCHAR(255),\
        address VARCHAR(255) \
    )`)


    console.log(`Created patient table successfully.`)

    // Ideally these would all be together. 
    const alter_table_command = `ALTER TABLE ${PATIENT_TABLE_NAME} ADD CONSTRAINT patients_key PRIMARY KEY (nin)`;
    await client.query(alter_table_command);
    console.log("added primary key in patient table.")
}

async function drop_all_tables_if_necessary(form_names) {
    // drop this table.

    if (form_names.length == 0) {
        console.log(`Nothing to drop!`)
    }

    for (i = 0; i < form_names.length; i++) {
        let form = form_names[i].form_name
        console.log(`Dropping ${form}`)
        await client.query(`DROP TABLE IF EXISTS ${form}`);
        console.log(`Dropped ${form}`)
    }
}

// My Python roots.
async function set_up() {


    console.log(`Set up initiated.`)

    await client.connect();
    console.log(`Connected...`)
    console.log(``)

    let form_names = await get_all_user_created_forms();

    console.log(`Got User Created Forms`, form_names);

    console.log(`Dropping tables.`)
    await drop_all_tables_if_necessary(form_names);
    await drop_forms_table();

    await drop_form_metadata_table();

    await drop_patient_constraint();
    await drop_patient_table_if_necessary();


    console.log(``);
    console.log(`Creating tables.`)
    await create_form_metadata_table();
    
    

    // Patient biographical tables.
    await create_patient_table();
    console.log(`Patient Table Set up!`)

    await create_forms_table();
    console.log(`Created form connections successfully.`)

    console.log(``)

    await client.end();
    console.log(`Closed connection.`)
    console.log(`\nSuccess!`)
}

// Execute tasks here like in main
set_up()