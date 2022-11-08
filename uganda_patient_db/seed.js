const { Pool, Client } = require('pg');
const config = require("./config");
const faker = require('faker');

const PATIENT_TABLE_NAME = "patients" 


const client = new Client({
    user: config.db.user,
    host: config.db.host,
    password: config.db.password,
    port: config.db.port
})

// converts a patient object as row.
const patient = {
    asRow: function() {
        var row = `(${this.nin}, '${this.name}', ${this.outpatient_id_number}, '${this.birth_date}', ${this.sex},'${this.ward}','${this.phone_number}','${this.address}')`;
        return row;
    } 
};


const cervix_request_form = {
    asRow: function() {
        var row = `(${this.nin}, '${this.date}', '${this.via_result}', ${this.needs_biopsy}, '${this.map}','${this.notes}','${this.next_review_date}','${this.screening_staff_name}')`;
        return row;
    }
}


// MARK: Move to seed.js
async function seed_patient_table(min_nin, max_nin) {
    console.log("Attempting to seed patient table.")


    // Build array of rows we want to insert.
    // we are going to use let because arr is mutable and in Node.js, 
    // var gives a variable universal scope - what a stupid idea. 

    // rows will just be an array of strings with length max_nin - min_nin
    let rows = [
        
    ];

    // Build arr 
    for (i = min_nin; i < max_nin; i++) {
        const my_patient = create_patient(i); 
        const row = my_patient.asRow();
        rows.push(row);
    }
    

    // build query
    const query = `INSERT INTO ${PATIENT_TABLE_NAME}
     (nin, patient_name, outpatient_id_number, birth_date, sex, ward, phone_number, address)
     VALUES ${rows.join()}`


    // execute query
    await client.query(query)

}


async function clear_patients() {
    console.log(`Clearing Patients`);
    await client.query(`TRUNCATE TABLE patients;`)
    console.log(`Cleared seeded patients.`)
}

// helper function that generates a patient given a nin.
function create_patient(nin) {


    let names = ['sam', 'adam', "mitra", 'samson', 'eric', 'bobby', 'sean', 'hank', 'mickey', 'ada']
    // generate fake data and then assign to new patient object.
    const patient_name = names[nin % names.length]; 
    // console.log(patient_name)
    const outpatient_id_number = nin + 10;
    // const birth_date = faker.datatype.datetime().toUTCString();
    const sex = nin % 2;
    const ward = `${nin % 4} Ward`

    const phone_number = "4138675309";
    const address = `${patient_name} Street`

    const my_patient = Object.create(patient);
    my_patient.nin = nin
    my_patient.name = patient_name;
    my_patient.birth_date = `Wed, 14 Jun 2017 07:00:00 GMT`

    my_patient.phone_number = phone_number;
    my_patient.address = address;
    my_patient.ward = ward;
    my_patient.sex = sex;

    my_patient.outpatient_id_number = outpatient_id_number;
    return my_patient;
}

async function seed() {

    await client.connect();
    

    console.log(`clearing patients...`)
    await clear_patients();

    console.log(`seeding...`)
    await seed_patient_table(0, 10);

    console.log(`DONE... seeded patient table with simulated data.`)

    await client.end();
    console.log(`Safely Disconnected.`)
}

seed()

