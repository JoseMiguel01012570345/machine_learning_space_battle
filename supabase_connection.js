
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_API_KEY= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdybnV4bGNydmFka2J1eXdiYmhxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTk5OTE1MDgsImV4cCI6MjAzNTU2NzUwOH0.5tfHGHqJSjeE_VjAO89kKf3R67YL0S2TlTAzafR2pEw"
const SUPABASE_URL="https://grnuxlcrvadkbuywbbhq.supabase.co"

const supabase = createClient(SUPABASE_URL, SUPABASE_API_KEY)

const read_data = async ( table , fild ) => { 
    
    // fild: example -> "id" 
    // table : example -> "components"

    let {data , error} = await supabase.from(table).select(fild)
    
    if (error)
        console.log("an error has raised: ",error)
    else
        console.log( "data is: ", data) 
        
}

const insert_data = async ( data , table ) => {

    // data : example -> { id : "lada" , doors: 4 , wheels: 4 }
    // table : example -> "componets"

    let {status} = await supabase.from(table).insert(data)
    console.log("status:",status)

}

const delete_data = async(id_to_remove , table)=> {

    // id_to_update : example -> { id : "mazda" }
    // table : example -> "components"
    
    let {status} = await supabase.from(table).delete().match(id_to_remove)
    console.log("status:",status)

}


const update_data = async(id_to_update , table , new_data)=> { 
    
    // id_to_update : example -> { id : "mazda" }
    // table : example -> "components"
    // new_data : example -> { id: "ferrari" , door: 4 , roof: true }

    let {status} = await supabase.from(table).update( new_data ).match(id_to_update)
    console.log("status:",status)

}

update_data({ id :"mazda" }, "branch" , { id: "mercedes-benz" } )


