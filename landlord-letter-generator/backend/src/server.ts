// filepath: /home/helena/RIP_Landlords/landlord-letter-generator/backend/src/server.ts
import express from 'express';
import cors from 'cors';
import { Request, Response } from 'express';

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

interface LetterRequest {
  tenantName: string;
  address: string;
  issueDate: string;
  description: string;
}

app.post('/api/generate-letter', (req: Request<{}, {}, LetterRequest>, res: Response) => {
  const { tenantName, address, issueDate, description } = req.body;
  
  // Basic letter template
  const letter = `
Berlin, den ${issueDate}

Sehr geehrte Damen und Herren,

hiermit möchte ich, ${tenantName}, wohnhaft in ${address}, Sie darüber informieren, dass in meiner Wohnung folgende Mängel aufgetreten sind:

${description}

Ich bitte Sie, die Beseitigung dieser Mängel möglichst umgehend zu veranlassen.

Mit freundlichen Grüßen
${tenantName}
`;

  res.json({ letter });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});